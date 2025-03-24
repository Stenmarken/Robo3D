source config.sh

echo "Generating heavy fog distortions"
CUDA_VISIBLE_DEVICES=0 python fog_simulation.py \
            --root_folder $ROOT_FOLDER \
            --dst_folder  $HEAVY_FOG_DST \
            --inte_folder integral_lookup_tables_seg_heavy_0.2beta \
            --beta  0.2

echo "Generating moderate fog distortions"
python fog_simulation.py \
            --root_folder $ROOT_FOLDER \
            --dst_folder  $MODERATE_FOG_DST \
            --inte_folder  integral_lookup_tables_seg_moderate_0.05beta \
            --beta  0.05

echo "Generating light fog distortions"
python fog_simulation.py \
            --root_folder $ROOT_FOLDER  \
            --dst_folder  $LIGHT_FOG_DST \
            --inte_folder  integral_lookup_tables_seg_light_0.008beta \
            --beta  0.008

