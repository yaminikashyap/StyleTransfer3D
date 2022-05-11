DATA_DIR=""
RESULTS_DIR=""
GENERATOR_NORM="bn"
DISCRIMINATOR_NORM="bn"
NUM_LAYERS=2
W_CHAMFER=1.0
W_CYCLE_CHAMFER=0.1
W_ADVERSARIAL=1.0
W_PERCEPTUAL=0.0
W_CONTENT_REC=0.1
W_STYLE_REC=0.1
NUMBER_POINTS=2500
BATCH_SIZE=4
GEN_LR=0.001
DIS_LR=0.004
NEPOCH=180

python -u src/train.py \
--data_dir=$DATA_DIR \
--family "chair" \
--class_0 "armchair" \
--class_1 "straight chair,side chair" \
--batch_size=$BATCH_SIZE \
--weight_chamfer=$W_CHAMFER \
--weight_cycle_chamfer=$W_CYCLE_CHAMFER \
--weight_adversarial=$W_ADVERSARIAL \
--weight_perceptual=$W_PERCEPTUAL \
--weight_content_reconstruction=$W_CONTENT_REC \
--weight_style_reconstruction=$W_STYLE_REC \
--nepoch=$NEPOCH \
--generator_lrate=$GEN_LR \
--discriminator_lrate=$DIS_LR \
--best_results=$RESULTS_DIR \
