import tritonclient.grpc as grpcclient

class TritonModelGRPC:
    '''
        Sample model-request triton-inference-server with gRPC
    '''
    def __init__(self,
                triton_host = 'localhost:8001', # default gRPC port
                triton_model_name = 'wav2vec_general_v2',
                verbose = False):
        print('Init connection from Triton-inference-server')
        print('- Host: {}'.format(triton_host))
        print('- Model: {}'.format(triton_model_name))
        self.triton_host = triton_host
        self.triton_model_name = triton_model_name
        self.model = grpcclient.InferenceServerClient(url=self.triton_host,
                                                            verbose=verbose,
                                                            ssl=False,
                                                            root_certificates=None,
                                                            private_key=None,
                                                            certificate_chain=None)
        if not self.model.is_server_live():
            print("FAILED : is_server_live")
            sys.exit(1)

        if not self.model.is_server_ready():
            print("FAILED : is_server_ready")
            sys.exit(1)
        
        if not self.model.is_model_ready("wav2vec_general_v2"):
            print("FAILED : is_model_ready")
            sys.exit(1)
        self.verbose = verbose

    def run(self, feats):
        # Input shape must be [-1]
        assert len(feats.shape) == 2, "Shape not support: {}".format(feats.shape)
        assert feats.shape[0] == 1, "Shape not support: {}".format(feats.shape)
        feats_length = feats.shape[-1]
        if self.verbose:
            print('='*50)
            print('- Input shape: [1, {}]'.format(feats_length))
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput('input', [1, feats_length], "FP32"))
        inputs[0].set_data_from_numpy(feats)
        outputs.append(grpcclient.InferRequestedOutput('output'))
        if self.verbose:
            tik = time.time()
        results = self.model.infer(
            model_name="wav2vec_general_v2",
            inputs=inputs,
            outputs=outputs,
            client_timeout=None)
        if self.verbose:
            tok = time.time()
            print('- Time cost:', tok - tik)
        output = results.as_numpy('output')
        return output

