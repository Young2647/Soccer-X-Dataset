from optparse import OptionParser

def get_args():
    parser = OptionParser()
    parser.add_option('--name', type=str, default='debug', help='name')
    parser.add_option('--epochs', default=10000, type='int',help='number of epochs')
    parser.add_option('--batch_size', default=128,type='int', help='batch size')
    parser.add_option('--lr', default=0.0001,type='float', help='learning rate') 
    parser.add_option('--dropout', default=0.3,type='float', help='dropout rate')
    parser.add_option('--device', default=0,type=int, help='which GPU device to use')
    parser.add_option('--dataset_path', type=str, default="dataset/pkl/traj.pkl", help='path to .pkl file')
    parser.add_option('--future_frame', type=int, default=45, help='future frames')
    parser.add_option('--past_frame', type=int, default=10, help='past frames')
    parser.add_option('--weight_decay', type=float, default=0.01, help='optim weight decay')
    parser.add_option('--save_interval', type=int, default=100, help='save interval')
    parser.add_option('--save_path', type=str, default='./checkpoints', help='save dir path')
    parser.add_option('--load_path', type=str, default=None, help='load path')
    parser.add_option('--test_path', type=str, default=rf'checkpoints\20240905\best.pth', help='test model path')
    parser.add_option('--onnx_path', type=str, default=rf'checkpoints\20240905\best.pth', help='export model path')
    
    # # gaussian diffusion
    # parser.add_option('--denoise_step', type=int, default=4, help='denoise step')
    # parser.add_option('--noise_schedule', type=str, default='cosine', help='linear, conine')
    # parser.add_option('--sigma_small', default=True, help='')


    (options, args) = parser.parse_args()
    return options