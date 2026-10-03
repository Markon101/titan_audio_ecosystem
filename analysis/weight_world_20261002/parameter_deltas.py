#!/usr/bin/env python3
"""Cheap parameter-group deltas; no SVD or useful-learning claim."""
import json
import sys
import numpy as np
import run_matrix as matrix
sys.path.insert(0,str(matrix.ROOT/'analysis/metastable_20261001'))
from checkpoint_geometry import SafeTensorStore, group


def compare(before_path,after_path):
    before=SafeTensorStore(before_path);after=SafeTensorStore(after_path);sums={}
    for name in before.header:
        if name=='__metadata__':continue
        a=np.asarray(before.array(name),dtype=float);b=np.asarray(after.array(name),dtype=float)
        g=group(name);item=sums.setdefault(g,[0.,0.,0])
        item[0]+=float(np.sum(a*a));item[1]+=float(np.sum((b-a)**2));item[2]+=a.size
    return {name:{'parameters':n,'relative_l2_change':float(np.sqrt(d/max(w,1e-24)))}
            for name,(w,d,n) in sums.items()}


def main():
    result={'schema':1,'scope':'net parameter movement does not establish beneficial learning',
        '703_to_58107':compare(matrix.artifact(matrix.E,'model','safetensors'),matrix.artifact(matrix.P,'model','safetensors')),
        '58107_to_60919':compare(matrix.artifact(matrix.P,'model','safetensors'),matrix.artifact(matrix.C,'model','safetensors'))}
    (matrix.HERE/'parameter_deltas.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({key:{name:round(value['relative_l2_change'],5) for name,value in result[key].items()
                          if name in ('gru','msfield','temporal_decoder','monitor_head')}
                     for key in ('703_to_58107','58107_to_60919')}))


if __name__=='__main__':main()
