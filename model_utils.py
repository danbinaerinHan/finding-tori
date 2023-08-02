def cal_conv_parameters(hparams, cnn_input_size):
    num_layers = hparams.num_layers
    parameters = [{'input_channel':0, 'output_channel':0, 'max_pool':0} for i in range(num_layers)]
    
    mean_pool_size = hparams.compression_ratio ** (1/ (num_layers-1))
    base_pool_size = int(mean_pool_size)
    num_longer_pool = int( (mean_pool_size - base_pool_size) * (num_layers-1))
    assert base_pool_size ** (num_layers-1-num_longer_pool) * (base_pool_size+1) ** (num_longer_pool) < hparams.compression_ratio

    for param in parameters[:num_longer_pool]:
        param['max_pool'] = base_pool_size + 1
    for param in parameters[num_longer_pool:-1]:
        param['max_pool'] = base_pool_size

    if hparams.use_gradual_size:
        for param in parameters[:num_layers//3]:
            param['output_channel'] = hparams.hidden_size // 2
            param['input_channel'] = hparams.hidden_size // 2
        for param in parameters[num_layers//3:num_layers*2//3]:
            param['output_channel'] = hparams.hidden_size
            param['input_channel'] = hparams.hidden_size
        for param in parameters[num_layers*2//3:]:
            param['output_channel'] = hparams.hidden_size * 2
            param['input_channel'] = hparams.hidden_size * 2
        parameters[num_layers//3]['input_channel'] = hparams.hidden_size // 2
        parameters[num_layers*2//3]['input_channel'] = hparams.hidden_size
    else:
        for param in parameters:
            param['output_channel'] = hparams.hidden_size
            param['input_channel'] = hparams.hidden_size
    
    parameters[0]['input_channel'] = cnn_input_size

    return parameters