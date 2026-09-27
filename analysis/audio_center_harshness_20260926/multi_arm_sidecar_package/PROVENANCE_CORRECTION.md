# Provenance correction for this package

The WAV files in this directory match their recorded SHA-256 output hashes.
The package's `receipt.json` must not be used to claim checkpoint or run
lineage. It associates a 28,798,976-frame, 599.9787-second source WAV
(`22091e4f6fceddd7b61e304bf6b9ddb0d03b98596365346f688cfd3591082185`)
with a copied metadata file reporting only 112 chunks, 458,752 frames,
9.5573 seconds, and end step 81,483. That metadata belongs to a later short
continuation. A matching original metadata snapshot for the long WAV was not
available to this audit.

The seven clips remain valid *audio derivatives* of the input paths and hashes
listed in the receipt. They are historical, non-blind candidates. Two named
"wide" candidates also include EQ, so comparisons with the legacy prime
change both source interval and filter. The original EQ copies were attenuated
by about 0.9-1.0 dB RMS, which further confounds unadjusted listening.

`scripts/prime_package.py` now checks rendered duration and frame count before
associating a run metadata file with a full WAV; it rejects this mismatch.
Use the corrected diagnostic report and newly level-matched listening panel
for comparisons. This note preserves the original receipt and files rather
than silently rewriting their history.
