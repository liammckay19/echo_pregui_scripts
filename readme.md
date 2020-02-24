# Echo pregui scripts 
Usage (windows): `run.sh [output directory name] [plate temperature: 4 or 20] [plate_id number] `
Usage (linux/unix): `bash run.sh [output directory name] [plate temperature: 4 or 20] [plate_id number] `

Multiple plates (windows): `run.sh [output directory name] [plate temperature: 4 or 20] [plate_id number] <plate_id number(s)...> `
Multiple plates (linux/unix): `bash run.sh [output directory name] [plate temperature: 4 or 20] [plate_id number] <plate_id number(s)...>  `

#### SAMPLE OUTPUT

```▶ bash run.sh output_test 20 10962

xray@169.230.29.115's password: 

selected IDs: 108228 105546

After entering password, should only take ~30 sec to download all images on UCSFwpa

xray@169.230.29.115's password: 



organizing images.

('output_test/organizedWells', 'already exists. continuing')

overlaying images.

()

53%|█████▎    | 51/96 [00:09<00:08,  5.46it/s]^C




