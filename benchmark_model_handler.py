# Record the running time
import time
import os
import glob
import pandas as pd
import numpy as np
import logging
# Inbuilt or External Modules
import argparse # customized arguments in .bash
from preprocess_scripts.data_loader import TSFDataLoader
import benchmarks
import tensorflow as tf
from causalimpact import CausalImpact

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def sMAPE_tf(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    smape_values = tf.abs(y_pred - y_true) / (tf.abs(y_pred) + tf.abs(y_true)) * 2
    smape_per_series = tf.reduce_mean(smape_values, axis=1)
    mean_smape = tf.reduce_mean(smape_per_series)

    return mean_smape  # Convert TensorFlow tensor to NumPy array for compatibility


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser("Train different forecasting models")
    # basic config
    argument_parser.add_argument('--seed', type=int, default=1234, help='random seed')
    argument_parser.add_argument(
        '--model',
        type=str,
        default='CausalImpact',
        choices=[
          'DeepProbCP',
          'ArCo',
          'CausalImpact',
          'MQRNN',
          'DeepAR',
          'MQRNN-local',
          'FFNN',
          'RNN',
          'TFT',
          'tsmixer',
          'tsmixer_rev_in',
          'TimeGPT'
      ],
        help='model name, options: [tsmixer, tsmixer_rev_in]',
    )
    argument_parser.add_argument(
        '--dataset_name',
        default='calls911_benchmarks',
        help='Unique string for the name of the dataset')
    argument_parser.add_argument(
      '--feature_type',
      type=str,
      default='M',
      choices=['S', 'M', 'MS'],
      help=(
          'forecasting task, options:[M, S, MS]; M:multivariate predict'
          ' multivariate, S:univariate predict univariate, MS:multivariate'
          ' predict univariate'
      ),
  )
    argument_parser.add_argument(
        '--target',
        type=str, default='ABINGTON', help='target feature in S or MS task'
  )
    argument_parser.add_argument(
        '--treated',
        type=list,
        default=["ABINGTON",  "AMBLER",  "CHELTENHAM",  "COLLEGEVILLE",  "CONSHOHOCKEN", 
                   "EAST GREENVILLE",  "EAST NORRITON",  "FRANCONIA" , "GREEN LANE", "HATFIELD TOWNSHIP", 
                   "HORSHAM" , "JENKINTOWN",  "LANSDALE",  "LIMERICK",  "LOWER GWYNEDD", 
                   "LOWER MERION",  "LOWER MORELAND",  "LOWER POTTSGROVE",  "LOWER PROVIDENCE",  "LOWER SALFORD", 
                   "MARLBOROUGH",  "MONTGOMERY",  "NARBERTH",  "PENNSBURG",  "PERKIOMEN", 
                   "PLYMOUTH",  "POTTSTOWN",  "RED HILL",  "ROCKLEDGE",  "ROYERSFORD", 
                   "SCHWENKSVILLE",  "SKIPPACK",  "SOUDERTON",  "TELFORD",  "TOWAMENCIN", 
                   "UPPER DUBLIN",  "UPPER FREDERICK",  "UPPER GWYNEDD",  "UPPER HANOVER",  "UPPER MERION", 
                   "UPPER MORELAND",  "UPPER POTTSGROVE",  "UPPER PROVIDENCE",  "UPPER SALFORD",  "WEST CONSHOHOCKEN", 
                   "WEST NORRITON",  "WEST POTTSGROVE",  "WHITEMARSH",  "WHITPAIN",  "WORCESTER"],
        help='treated group'
  )
    argument_parser.add_argument(
        '--control',
        type=list,
        default=["BRIDGEPORT", "BRYN ATHYN", "DOUGLASS", "HATBORO", "HATFIELD BORO", 
                   "LOWER FREDERICK", "NEW HANOVER", "NORRISTOWN", "NORTH WALES", "SALFORD", 
                   "SPRINGFIELD", "TRAPPE"],
        help='control group'
  )
    argument_parser.add_argument(
      '--checkpoint_dir',
      type=str,
      default='./checkpoints/',
      help='location of model checkpoints',
    )
    argument_parser.add_argument(
        '--delete_checkpoint',
        action='store_true',
        help='delete checkpoints after the experiment',
    )
    argument_parser.add_argument(
        '--no_of_series', # ff_dim in txmiser
        default=62,
        help='The number of series in the dataset')
    
    argument_parser.add_argument(
        '--input_size',
        required=False, # seq_len in txmiser
        default=16,
        help='The input size of the moving window. Default is 0')
    argument_parser.add_argument( # pred_len in txmiser
        '--forecast_horizon',
        default=7,
        help='The forecast horizon of the dataset')
    
    # model hyperparameter
    argument_parser.add_argument(
        '--n_block',
        type=int,
        default=2,
        help='number of block for deep architecture',
    )
    argument_parser.add_argument(
        '--dropout',
        type=float, 
        default=0.05, 
        help='dropout rate'
    )
    argument_parser.add_argument(
        '--norm_type',
        type=str,
        default='B',
        choices=['L', 'B'],
        help='LayerNorm or BatchNorm',
    )
    argument_parser.add_argument(
        '--activation',
        type=str,
        default='relu',
        choices=['relu', 'gelu'],
        help='Activation function',
    )
    argument_parser.add_argument(
        '--kernel_size',
        type=int, 
        default=4, 
        help='kernel size for CNN'
    )
    argument_parser.add_argument(
        '--temporal_dim', 
        type=int, 
        default=16, 
        help='temporal feature dimension'
    )
    argument_parser.add_argument(
        '--hidden_dim', 
        type=int, 
        default=25, 
        help='hidden feature dimension'
    )
    
    # optimization
    argument_parser.add_argument(
        '--num_workers', 
        type=int, 
        default=3, 
        help='data loader num workers'
    )
    argument_parser.add_argument(
        '--train_epochs', 
        type=int, 
        default=100, 
        help='train epochs'
    )
    argument_parser.add_argument(
        '--batch_size', 
        type=int, 
        default=5, 
        help='batch size of input data'
    )
    argument_parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help='optimizer learning rate',
    )
    argument_parser.add_argument(
        '--patience', 
        type=int, 
        default=5, 
        help='number of epochs to early stop'
    )

    # save results # need to change!!!
    argument_parser.add_argument(
        '--result_path', 
        default='results/', 
        help='path to save result'
    )
    
    argument_parser.add_argument(
        '--optimizer',
        default='adam', # used in tsmixer
        choices=['cocob',
                'adagrad'],
        help='The type of the optimizer(cocob/adam/adagrad...). Default is cocob')
    
    argument_parser.add_argument(
        '--initial_hyperparameter_values_file',
        help='The file for the initial hyperparameter configurations')

    argument_parser.add_argument(
        '--quantile_range', # not all need to infer quantile distr.
        required=False, 
        help='The range of the quantile for quantile forecasting. Default is np.linspace(0, 1, 21)')
    argument_parser.add_argument(
        '--evaluation_metric',
        default='sMAPE',
        choices=['mae',
                'mse'],
        help='The evaluation metric like sMAPE. Default is CRPS')
    argument_parser.add_argument(
        '--loss',
        default='mae',
        choices=['mse'],
        help='The evaluation metric like sMAPE. Default is CRPS')
    argument_parser.add_argument('--without_stl_decomposition',
                                 required=False,
                                 help='Whether not to use stl decomposition(0/1). Default is 0')

    # parse the user arguments
    args = argument_parser.parse_args()

    if 'tsmixer' in args.model:
        exp_id = f'{args.dataset_name}_{args.feature_type}_{args.model}_i{args.input_size}_h{args.forecast_horizon}_lr{args.learning_rate}_nt{args.norm_type}_{args.activation}_nb{args.n_block}_dp{args.dropout}_n{args.no_of_series}'
    elif 'CausalImpact' in args.model:
        exp_id = f'{args.dataset_name}_{args.feature_type}_{args.model}_h{args.forecast_horizon}_fd{args.no_of_series}'
    elif 'DeepProbCP' in args.model:
        exp_id = f'{args.dataset_name}_{args.feature_type}_{args.model}_i{args.input_size}_h{args.forecast_horizon}_fd{args.no_of_series}'
    else:
        raise ValueError(f'Unknown model type: {args.model}')

    # load datasets
    data_row = pd.read_csv('./datasets/text_data/EMS-MC/'+args.dataset_name+'.csv')
    
    print(args.model)
    # train model
    if 'tsmixer' in args.model:
        data_loader = TSFDataLoader(
            args.dataset_name,
            args.batch_size,
            args.input_size,
            args.forecast_horizon,
            args.feature_type,
            args.target,
        )
        train_data = data_loader.get_train()
        val_data = data_loader.get_val()
        test_data = data_loader.get_test()

        build_model = getattr(benchmarks, args.model).build_model
        model = build_model(
            input_shape=(args.input_size, data_loader.n_feature),
            pred_len=args.forecast_horizon,
            norm_type=args.norm_type,
            activation=args.activation,
            dropout=args.dropout,
            n_block=args.n_block,
            ff_dim=args.no_of_series,
            target_slice=data_loader.target_slice,
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        if args.evaluation_metric=='sMAPE':
            model.compile(optimizer=optimizer, loss=args.loss, metrics=[sMAPE_tf])
        else:
            model.compile(optimizer=optimizer, loss=args.loss, metrics=[args.evaluation_metric])

        checkpoint_path = os.path.join(args.checkpoint_dir, f'{exp_id}_best')
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
        )
        early_stop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=args.patience
        )
        start_training_time = time.time()
        history = model.fit(
            train_data,
            epochs=args.train_epochs,
            validation_data=val_data,
            callbacks=[checkpoint_callback, early_stop_callback],
        )
        end_training_time = time.time()
        elasped_training_time = end_training_time - start_training_time
        print(f'Training finished in {elasped_training_time} secconds')

        # evaluate best model
        best_epoch = np.argmin(history.history['val_loss'])
        model.load_weights(checkpoint_path)
        test_result = model.evaluate(test_data)
        test_smape = test_result[1]
        if args.delete_checkpoint:
            for f in glob.glob(checkpoint_path + '*'):
                os.remove(f)
    elif 'CausalImpact' in args.model:
        start_training_time = time.time()
        smape_values_per_series = []
        for i in args.treated:
            ci = CausalImpact(data_row.loc[:,[i]+args.control],
                 [0,len(data_row['date'])-args.forecast_horizon],
                 [len(data_row['date'])-args.forecast_horizon+1,
                  len(data_row['date'])-1])
            # evaluate the model
            y_pred = ci.inferences.loc[(len(data_row['date'])-args.forecast_horizon+1):(len(data_row['date'])-1),'preds']
            y_true = ci.post_data.iloc[:,0]
            smape_values = (np.abs(y_pred - y_true) /
                    (np.abs(y_pred) + np.abs(y_true))) * 2
            smape_values_per_series.append(np.mean(smape_values))
        test_smape = np.mean(smape_values_per_series)
        end_training_time = time.time()
        elasped_training_time = end_training_time - start_training_time
        print(f'Training finished in {elasped_training_time} secconds')
        args.norm_type = None
        args.activation = None
        args.n_block = None
        args.dropout = None
        args.input_size = None
        args.learning_rate = None
    elif 'DeepProbCP' in args.model:
        start_training_time = time.time()

        end_training_time = time.time()
        elasped_training_time = end_training_time - start_training_time
        print(f'Training finished in {elasped_training_time} secconds')


        LSTM_USE_PEEPHOLES = True # LSTM with “peephole connections"
        BIAS = False # in tf.keras.layers.dense

        # define the key word arguments for the different model types
        model_kwargs = {
            'use_bias': BIAS,
            'use_peepholes': LSTM_USE_PEEPHOLES,
            'input_size': args.input_size,
            'output_size': args.forecast_horizon,
            'optimizer': args.optimizer,
            'quantile_range': args.quantile_range,
            'evaluation_metric': args.evaluation_metric,
            'no_of_series': args.no_of_series,
            'seed': args.seed,
            'cell_type': 'LSTM',
            'without_stl_decomposition': 1
        }

        # select the model type
        model = StackingModel(**model_kwargs)
    else:
        raise ValueError(f'Model not supported: {args.model}')

    

    # save result to csv
    data = {
        'data': [args.dataset_name],
        'model': [args.model],
        'input_size': [args.input_size],
        'forecast_horizon': [args.forecast_horizon],
        'lr': [args.learning_rate],
        # 'mae': [test_result[0]],
        'smape': [test_smape],
        # 'val_mae': [history.history['val_loss'][best_epoch]],
        # 'val_smape': [history.history['val_smape'][best_epoch]],
        # 'train_mae': [history.history['loss'][best_epoch]],
        # 'train_smape': [history.history['smape'][best_epoch]],
        'training_time': elasped_training_time,
        'norm_type': args.norm_type,
        'activation': args.activation,
        'n_block': args.n_block,
        'dropout': args.dropout,
        'no_of_series': args.no_of_series
    }

    df = pd.DataFrame(data)
    if os.path.exists(args.result_path+'result.csv'):
        df.to_csv(args.result_path+'result.csv', mode='a', index=False, header=False)
    else:
        df.to_csv(args.result_path+'result.csv', mode='w', index=False, header=True)