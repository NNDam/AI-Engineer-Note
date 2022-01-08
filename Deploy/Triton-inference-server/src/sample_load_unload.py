'''
Example
    - Load model
        python3 sample_load_unload.py --models emotion_recognition_v1.1
    - Unload model
        python3 sample_load_unload.py --unload --models emotion_recognition_v1.1 
    - Load model from file
        python3 sample_load_unload.py --path --models model_list.txt
'''
import argparse
import tritonclient.grpc as grpcclient

parser = argparse.ArgumentParser(description='Load/Unload model')
parser.add_argument('--models', default="", help='list of model names to load/unload')
parser.add_argument('--unload', action = "store_true", help='load or unload model') 
parser.add_argument('--reload', action = "store_true", help='reload model') 
parser.add_argument('--path', action = "store_true", help='get list of models from filepath') 
parser.add_argument('--url', default="localhost:8001", help='default triton-server URL')
args = parser.parse_args()

if not args.path:
    MODEL_NAMES = args.models.strip().split(',')
else:
    MODEL_NAMES = open(args.models).read().strip('\n').split('\n')
URL = args.url
triton_client = grpcclient.InferenceServerClient(url=URL, verbose=True)
triton_client.is_server_live()
triton_client.get_model_repository_index().models
if args.unload:
    for MODEL_NAME in MODEL_NAMES:
        if triton_client.is_model_ready(MODEL_NAME):
            print('UNLOAD: {}'.format(MODEL_NAME))
            triton_client.unload_model(MODEL_NAME)
        else:
            print('Skip: {}'.format(MODEL_NAME))
else:
    for MODEL_NAME in MODEL_NAMES:
        if triton_client.is_model_ready(MODEL_NAME):
            if args.reload:
                print('RELOAD: {}'.format(MODEL_NAME))
                triton_client.unload_model(MODEL_NAME)
                triton_client.load_model(MODEL_NAME)
            else:
                print('Skip: {}'.format(MODEL_NAME))
        else:
            print('LOAD: {}'.format(MODEL_NAME))
            triton_client.load_model(MODEL_NAME)


print('='*70)
triton_client.get_model_repository_index().models