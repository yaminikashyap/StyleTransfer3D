import argparse

def parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--family', type=str)
    parser.add_argument('--class_0', type=str)
    parser.add_argument('--class_1', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--num_layers_style', type=int)
    parser.add_argument('--weight_chamfer', type=float)
    parser.add_argument('--weight_cycle_chamfer', type=float)
    parser.add_argument('--weight_adversarial', type=float)
    parser.add_argument('--weight_perceptual', type=float)
    parser.add_argument('--weight_content_reconstruction', type=float)
    parser.add_argument('--weight_style_reconstruction', type=float)
    parser.add_argument('--nepoch', type=int)
    parser.add_argument('--generator_lrate', type=float)
    parser.add_argument('--discriminator_lrate', type=float)
    parser.add_argument('--best_results_dir', type=str)

    opt = parser.parse_args()
    
    return opt
