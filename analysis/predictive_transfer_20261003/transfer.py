"""Persisted, fixed predictive-code bank. No fitting on evaluation trajectories."""
import sys
from pathlib import Path
import json
import numpy as np

PREVIOUS = Path(__file__).resolve().parents[1] / 'predictive_density_20261003'
sys.path.insert(0, str(PREVIOUS))
from predictive import Quantizer, projection, fit, forecast_indices, summaries, COEFFICIENT_BITS


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def scaler(values):
    return {'mean': values.mean(0), 'scale': np.maximum(values.std(0), 1e-6),
            'minimum': values.min(0), 'maximum': values.max(0)}


def standardize(values, state):
    return np.clip((values-np.asarray(state['mean'])) / np.asarray(state['scale']), -30, 30)


def delayed(y, lags):
    return np.concatenate([y if lag == 0 else np.concatenate((np.repeat(y[:1],lag,axis=0),
                                                             y[:-lag]),axis=0) for lag in lags],axis=1)


class FrozenBank:
    def __init__(self, state):
        self.state = state
        self.quantizer = Quantizer.__new__(Quantizer)
        self.quantizer.edges = [np.asarray(v) for v in state['quantizer']['edges']]
        self.quantizer.marginals = [np.asarray(v) for v in state['quantizer']['marginals']]

    def save(self, path):
        with Path(path).open('x') as output:
            output.write(json.dumps(plain(self.state),sort_keys=True,allow_nan=False)+'\n')

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def features(self, y, views, warmup=0):
        s = self.state
        yn = standardize(np.asarray(y,dtype=np.float64),s['y_scaler'])
        if not np.isfinite(yn).all():
            raise ValueError('nonfinite target')
        features = {}
        for name in ('history','slow'):
            specification = s['history'][name]
            raw = delayed(yn,specification['lags']) @ np.asarray(specification['projection'])
            features[name] = standardize(raw,specification['scaler'])
        clock = (np.arange(len(y))+warmup/s['sample_chunks']-s['train_end']/2)/s['train_end']
        features['history'] = np.column_stack((features['history'],clock))
        features['output_slow'] = np.column_stack((features['history'],features.pop('slow')))
        for name, specification in s['views'].items():
            value = np.asarray(views[name],dtype=np.float64)
            if len(value) != len(y) or not np.isfinite(value).all():
                raise ValueError('view shape/nonfinite')
            raw = standardize(value,specification['before']) @ np.asarray(specification['projection'])
            features[name] = np.column_stack((features['history'],standardize(raw,specification['after'])))
        history_width = features['history'].shape[1]
        for left,right in s['pairs']:
            features[left+'+'+right] = np.column_stack((features['history'],features[left][:,history_width:],
                                                       features[right][:,history_width:]))
        return yn,features

    def predictions(self, y, views, horizon, warmup=0, include_padding=False):
        yn,x = self.features(y,views,warmup)
        h = int(horizon)
        ids = np.arange(0 if include_padding else self.state['history_start'],len(y)-h)
        specification = self.state['horizons'][str(h)]
        predictions = {}
        for name,model in specification['models'].items():
            kind = model['kind']
            if kind == 'iid':
                continue
            if kind == 'linear':
                values = np.column_stack((np.ones(len(ids)),x[name][ids])) @ np.asarray(model['weights'])
            elif kind == 'persistence':
                values = yn[ids]
            elif kind == 'trend':
                values = yn[ids]+h*(yn[ids]-yn[np.maximum(ids-1,0)])
            else:
                period = model['period']
                cycles = (h-1)//period+1
                values = yn[np.maximum(ids-(cycles*period-h),0)].copy()
                if kind == 'affine':
                    for _ in range(cycles):
                        values = values*np.asarray(model['slope'])+np.asarray(model['offset'])
            predictions[name] = values
        return ids,yn[ids+h],predictions

    @classmethod
    def train(cls, y, views=None, horizons=(1,4,16,64), sample_chunks=1):
        y = np.asarray(y,dtype=np.float64)
        views = views or {}
        n,d = y.shape
        end,calend = int(n*.4),int(n*.6)
        state = {'schema':1,'numpy_version':np.__version__,'sample_chunks':sample_chunks,
                 'train_end':end,'calibration_end':calend,'samples':n,
                 'y_scaler':scaler(y[:end]),'history':{},'views':{},'pairs':[],'horizons':{}}
        yn = standardize(y,state['y_scaler'])
        q = Quantizer(yn[:end])
        state['quantizer'] = {'edges':q.edges,'marginals':q.marginals}
        for name,lags,width,tag in [('history',(0,1,4,16) if sample_chunks==1 else (0,1,2,4),8,'short_history'),
                                    ('slow',(32,64,128) if sample_chunks==1 else (8,16),4,'slow_history')]:
            raw = delayed(yn,lags)
            matrix = projection(np.eye(raw.shape[1]),width,tag)
            state['history'][name] = {'lags':lags,'projection':matrix,'scaler':scaler((raw@matrix)[:end])}
        state['history_start'] = max(state['history']['slow']['lags'])
        for name,value in views.items():
            value = np.asarray(value,dtype=np.float64)
            before = scaler(value[:end])
            matrix = projection(np.eye(value.shape[1]),4,name)
            raw = standardize(value,before)@matrix
            state['views'][name] = {'before':before,'projection':matrix,'after':scaler(raw[:end])}
        for left,right in [('gru','field_coarse'),('host','field_coarse'),
                           ('morphic_delta_l01','morphic_delta_l16'),('motif','gru')]:
            if left in views and right in views:
                state['pairs'].append((left,right))
        bank = cls(state)
        _,features = bank.features(y,views)
        for h in horizons:
            ids,masks = forecast_indices(n,h,state['history_start'])
            actual = yn[ids+h]
            models = {'iid':{'kind':'iid','parameters':0}}
            def add(name,kind,prediction,parameters,**extra):
                residual = actual[masks['calibration']]-prediction[masks['calibration']]
                models[name] = {'kind':kind,'parameters':parameters,'scale':np.maximum(np.sqrt(np.mean(residual**2,axis=0)),.15),**extra}
            add('persistence','persistence',yn[ids],d)
            add('trend','trend',yn[ids]+h*(yn[ids]-yn[ids-1]),d)
            periods = [4,16,64,128] if sample_chunks==1 else [2,4,8,16]
            limit = end//3
            periods.append(min(range(1,limit+1),key=lambda p:float(np.mean((yn[p:end]-yn[:end-p])**2))))
            def affine(p):
                past,current = yn[:end-p],yn[p:end]
                pm,cm = past.mean(0),current.mean(0)
                slope = np.sum((past-pm)*(current-cm),axis=0)/np.maximum(np.sum((past-pm)**2,axis=0),1e-8)
                offset = cm-slope*pm
                return slope,offset,float(np.mean((current-past*slope-offset)**2))
            affine_periods = [p for p in periods if p<=end-8]
            affine_periods.append(min(range(1,limit+1),key=lambda p:affine(p)[2]))
            for p in sorted(set(periods)):
                lag = ((h-1)//p+1)*p-h
                add(f'period_{p*sample_chunks}','copy',yn[np.maximum(ids-lag,0)],d+1,period=p)
            for p in sorted(set(affine_periods)):
                slope,offset,_ = affine(p)
                cycles = (h-1)//p+1
                prediction = yn[np.maximum(ids-(cycles*p-h),0)].copy()
                for _ in range(cycles):prediction = prediction*slope+offset
                add(f'affine_period_{p*sample_chunks}','affine',prediction,3*d+1,period=p,slope=slope,offset=offset)
            for name,value in features.items():
                model = fit(value[ids],actual,masks['train'],masks['calibration'])
                models[name] = {'kind':'linear','weights':model.weights,'scale':model.scale,'parameters':model.parameters}
            specification = {'models':models,'train_rows':int(masks['train'].sum()),'calibration_rows':int(masks['calibration'].sum())}
            state['horizons'][str(h)] = specification
            _,_,predictions = bank.predictions(y,views,h,include_padding=True)
            codes = {name:q.iid_bits(actual) if name=='iid' else q.gaussian_bits(actual,predictions[name],model['scale'])
                     for name,model in models.items()}
            competitors = [name for name in models if name not in features or name=='history']
            calibration = {name:float(codes[name][masks['calibration']].sum()) for name in competitors}
            specification['comparator'] = min(competitors,key=lambda name:(calibration[name]+16*models[name]['parameters'],name))
            specification['unpenalized_comparator'] = min(competitors,key=lambda name:(calibration[name],name))
            specification['calibration_code_bits'] = calibration
        return bank

    def score(self,y,views=None,warmup=0,start=None,adapt_sigma=False):
        views = views or {}
        y = np.asarray(y,dtype=np.float64)
        state = self.state
        result = {'sample_chunks':state['sample_chunks'],'samples':len(y),'horizons':{},'adapt_sigma':adapt_sigma}
        transformed = (y-np.asarray(state['y_scaler']['mean']))/np.asarray(state['y_scaler']['scale'])
        result['target_scaler_clip_fraction'] = float(np.mean(np.abs(transformed)>30))
        result['target_outside_source_range_fraction'] = float(np.mean((y<np.asarray(state['y_scaler']['minimum']))|
                                                                      (y>np.asarray(state['y_scaler']['maximum']))))
        for hs,specification in state['horizons'].items():
            h = int(hs)
            ids,actual,predictions = self.predictions(y,views,h,warmup)
            boundary = 256//state['sample_chunks']
            calibration = ids+h<boundary
            mask = ids>=(boundary if adapt_sigma else (start if start is not None else state['history_start']))
            if mask.sum()<8 or (adapt_sigma and calibration.sum()<8):
                result['horizons'][str(h*state['sample_chunks'])] = {'supported':False,'reason':'insufficient purged calibration/audit pairs'}
                continue
            models = specification['models']
            scales = {name:(np.maximum(np.sqrt(np.mean((actual[calibration]-predictions[name][calibration])**2,axis=0)),.15)
                            if adapt_sigma else np.asarray(model['scale'])) for name,model in models.items() if name!='iid'}
            codes = {name:self.quantizer.iid_bits(actual) if name=='iid' else self.quantizer.gaussian_bits(actual,predictions[name],scales[name])
                     for name in models}
            history_scale = scales['history']
            common = {name:self.quantizer.gaussian_bits(actual,prediction,history_scale) for name,prediction in predictions.items()}
            item = {'supported':True,'forecasts':int(mask.sum()),'source_comparator':specification['comparator'],
                    'source_unpenalized_comparator':specification['unpenalized_comparator'],'probes':{},'synergy':{}}
            for name,model in models.items():
                if model['kind']!='linear':continue
                base = specification['comparator']
                entry = summaries(codes[base][mask]-codes[name][mask],model['parameters']-models[base]['parameters'],
                                  len(model['weights'])-1,y.shape[1],specification['train_rows'])
                for comparator,key in [('iid','iid_gain_bits'),('history','over_history_bits'),
                                       ('output_slow','over_same_capacity_slow_history_bits'),
                                       (specification['unpenalized_comparator'],'over_unpenalized_baseline_bits')]:
                    entry[key] = float((codes[comparator][mask]-codes[name][mask]).sum())
                entry['mean_only_common_history_sigma_gain_bits'] = float((common['history'][mask]-common[name][mask]).sum())
                entry['per_target_over_history_bits'] = (codes['history'][mask]-codes[name][mask]).sum(0).tolist()
                entry['per_target_common_sigma_gain_bits'] = (common['history'][mask]-common[name][mask]).sum(0).tolist()
                entry['scaled_mse'] = float(np.mean((actual[mask]-predictions[name][mask])**2))
                entry['sigma_rms'] = float(np.sqrt(np.mean(scales[name]**2)))
                item['probes'][name] = entry
            for left,right in state['pairs']:
                name=left+'+'+right
                gains = {key:item['probes'][key]['over_history_bits'] for key in (left,right,name)}
                item['synergy'][name] = {'history_conditional_joint_minus_sum_bits':gains[name]-gains[left]-gains[right],
                                         'joint_beyond_best_individual_bits':gains[name]-max(gains[left],gains[right]),'not_PID':True}
            result['horizons'][str(h*state['sample_chunks'])] = item
        return result
