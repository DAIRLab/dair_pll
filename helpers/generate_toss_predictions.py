"""Script to load a trained system then generate rollouts from the elbow toss
dataset.  Also saves a post-processed statistics file that has the position and
angle trajectory errors."""

import git
import os
import os.path as op
import pickle
import pdb
import torch
from torch import Tensor

from dair_pll import file_utils
from dair_pll.dataset_management import ExperimentDataManager
from dair_pll.deep_learnable_system import DeepLearnableSystemConfig
from dair_pll.drake_experiment import DrakeMultibodyLearnableExperiment, \
    DrakeDeepLearnableExperiment, MultibodyLearnableSystemConfig
from dair_pll.experiment import default_epoch_callback, TrainingState
from dair_pll.state_space import FloatingBaseSpace


PLL_TO_BAG_NUMBERS = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 13,
    13: 14,
    14: 15,
    15: 16,
    15: 17,
    16: 18,
    17: 19,
    18: 20,
    19: 21,
    20: 22,
    21: 23,
    22: 24,
    23: 25,
    24: 26,
    24: 27,
    25: 28,
    26: 29,
    27: 30,
    28: 31,
    29: 32,
    30: 33,
    30: 34,
    31: 35,
    32: 36,
    33: 37,
    34: 38,
    35: 39,
    36: 40,
    37: 41,
    38: 42,
    39: 43,
    40: 44,
    41: 45,
    42: 46,
    43: 47,
    44: 48,
    45: 49,
    46: 50,
    47: 51,
    48: 52,
    49: 53,
    49: 54,
    50: 55,
    51: 56,
    52: 57,
    53: 58,
    53: 59,
    53: 60,
    53: 61,
    54: 62,
    54: 63,
    54: 64,
    55: 65,
    56: 66,
    57: 67,
    58: 68,
    58: 69,
    59: 70,
    60: 71,
    61: 72,
    62: 73,
    63: 74,
    64: 75,
    64: 76,
    65: 77,
    66: 78,
    67: 79,
    68: 80,
    69: 81,
    70: 82,
    71: 83,
    72: 84,
    73: 85,
    74: 86,
    75: 87,
    76: 88,
    77: 89,
    78: 90,
    79: 91,
    80: 92,
    81: 93,
    82: 94,
    82: 95,
    82: 96,
    82: 97,
    83: 98,
    84: 99,
    85: 100,
    86: 101,
    87: 102,
    88: 103,
    89: 104,
    90: 105,
    91: 106,
    92: 107,
    93: 108,
    94: 109,
    95: 110,
    96: 111,
    97: 112,
    98: 113,
    99: 114,
    100: 115,
    101: 116,
    102: 117,
    103: 118,
    104: 119,
    105: 121,
    106: 122,
    107: 123,
    108: 124,
    109: 125,
    110: 126,
    111: 127,
    112: 128,
    113: 129,
    114: 130,
    115: 131,
    116: 132,
    117: 133,
    118: 134,
    119: 135,
    120: 136,
    121: 137,
    122: 138,
    123: 139,
    124: 140,
    125: 141,
    126: 142,
    126: 143,
    126: 144,
    126: 145,
    126: 146,
    127: 147,
    128: 148,
    129: 149,
    130: 150,
    131: 151,
    132: 152,
    133: 153,
    134: 154,
    135: 155,
    136: 156,
    137: 157,
    138: 158,
    139: 159,
    140: 160,
    141: 161,
    142: 162,
    143: 163,
    144: 164,
    145: 165,
    146: 166,
    147: 167,
    148: 168,
    148: 169,
    149: 170,
    150: 171,
    150: 172,
    151: 173,
    152: 174,
    153: 175,
    154: 176,
    154: 177,
    155: 178,
    156: 179,
    157: 180,
    158: 181,
    159: 182,
    160: 183,
    161: 184,
    162: 185,
    163: 186,
    164: 187,
    165: 188,
    166: 189,
    167: 190,
    168: 191,
    169: 192,
    170: 193,
    171: 194,
    172: 195,
    173: 196,
    174: 197,
    174: 198,
    175: 199,
    176: 200,
    177: 201,
    178: 202,
    179: 203,
    180: 204,
    181: 205,
    181: 206,
    181: 207,
    182: 208,
    183: 209,
    184: 210,
    185: 211,
    186: 212,
    187: 213,
    188: 214,
    189: 215,
    190: 216,
    190: 217,
    191: 218,
    192: 219,
    193: 220,
    194: 221,
    195: 222,
    196: 223,
    197: 224,
    198: 225,
    199: 226,
    200: 227,
    201: 228,
    202: 229,
    203: 230,
    203: 231,
    204: 232,
    205: 233,
    206: 234,
    207: 235,
    208: 236,
    209: 237,
    210: 238,
    211: 239,
    212: 240,
    213: 241,
    214: 242,
    215: 243,
    216: 244,
    217: 245,
    217: 246,
    218: 247,
    219: 248,
    220: 249,
    221: 250,
    222: 251,
    223: 252,
    224: 253,
    224: 254,
    224: 255,
    225: 256,
    226: 257,
    227: 258,
    228: 259,
    229: 260,
    229: 261,
    230: 262,
    231: 263,
    232: 265,
    233: 266,
    234: 267,
    235: 268,
    236: 269,
    237: 270,
    238: 271,
    239: 272,
    240: 273,
    241: 274,
    242: 275,
    242: 276,
    243: 277,
    244: 278,
    244: 279,
    244: 280,
    245: 281,
    246: 282,
    247: 283,
    248: 284,
    249: 285,
    250: 286,
    251: 287,
    252: 288,
    253: 289,
    253: 290,
    254: 291,
    254: 292,
    255: 293,
    256: 294,
    257: 295,
    258: 296,
    258: 297,
    259: 298,
    260: 299,
    261: 300,
    262: 301,
    263: 302,
    264: 303,
    265: 304,
    266: 305,
    266: 306,
    267: 307,
    268: 308,
    269: 309,
    270: 310,
    271: 311,
    272: 312,
    273: 313,
    274: 314,
    275: 315,
    276: 316,
    277: 317,
    277: 318,
    278: 319,
    279: 320,
    280: 321,
    281: 322,
    282: 323,
    283: 324,
    283: 325,
    284: 326,
    285: 327,
    286: 328,
    287: 329,
    288: 330,
    289: 331,
    290: 333,
    291: 334,
    292: 335,
    293: 336,
    294: 337,
    295: 338,
    296: 339,
    297: 340,
    298: 341,
    299: 342,
    300: 343,
    301: 344,
    302: 345,
    303: 346,
    304: 347,
    305: 348,
    306: 349,
    307: 350,
    308: 351,
    309: 352,
    310: 353,
    311: 354,
    312: 355,
    313: 356,
    314: 357,
    315: 358,
    316: 359,
    317: 360,
    318: 361,
    319: 362,
    320: 363,
    321: 364,
    322: 365,
    323: 366,
    324: 367,
    324: 368,
    325: 369,
    326: 370,
    327: 371,
    328: 372,
    329: 373,
    330: 374,
    331: 375,
    332: 376,
    332: 377,
    333: 378,
    334: 379,
    335: 380,
    336: 381,
    337: 382,
    338: 383,
    339: 384,
    339: 385,
    340: 386,
    341: 387,
    341: 388,
    342: 389,
    343: 390,
    344: 391,
    345: 392,
    346: 393,
    347: 394,
    347: 395,
    348: 396,
    349: 397,
    350: 398,
    351: 399,
    352: 400,
    353: 401,
    354: 402,
    355: 403,
    356: 404,
    357: 405,
    358: 406,
    359: 407,
    360: 408,
    361: 409,
    362: 410,
    363: 411,
    364: 412,
    365: 413,
    366: 414,
    367: 415,
    368: 416,
    369: 417,
    370: 418,
    371: 419,
    372: 420,
    373: 421,
    374: 422,
    375: 423,
    376: 424,
    377: 425,
    378: 426,
    379: 427,
    379: 428,
    380: 429,
    381: 430,
    382: 431,
    383: 432,
    384: 433,
    385: 434,
    386: 435,
    387: 436,
    388: 437,
    389: 438,
    390: 439,
    391: 440,
    392: 441,
    392: 442,
    393: 443,
    394: 444,
    395: 445,
    396: 446,
    397: 447,
    398: 448,
    399: 449,
    400: 450,
    401: 451,
    402: 452,
    403: 453,
    403: 454,
    404: 455,
    405: 456,
    406: 457,
    407: 458,
    408: 459,
    409: 460,
    410: 461,
    411: 462,
    412: 463,
    413: 464,
    414: 465,
    415: 466,
    416: 467,
    417: 468,
    418: 469,
    419: 470,
    420: 471,
    421: 472,
    422: 473,
    423: 474,
    424: 475,
    425: 476,
    426: 477,
    427: 478,
    428: 479,
    429: 480,
    430: 481,
    431: 482,
    432: 483,
    433: 484,
    433: 485,
    433: 486,
    434: 487,
    435: 488,
    436: 489,
    437: 490,
    438: 491,
    439: 492,
    440: 493,
    441: 494,
    442: 495,
    443: 496,
    444: 497,
    445: 498,
    446: 499,
    447: 500,
    447: 501,
    448: 502,
    449: 503,
    450: 504,
    451: 505,
    452: 506,
    453: 507,
    454: 508,
    454: 509,
    455: 510,
    456: 511,
    457: 512,
    458: 513,
    459: 514,
    460: 515,
    461: 516,
    462: 517,
    463: 518,
    464: 519,
    465: 520,
    466: 521,
    467: 522,
    468: 523,
    469: 524,
    470: 525,
    471: 526,
    472: 527,
    473: 528,
    474: 529,
    475: 530,
    476: 531,
    477: 532,
    478: 533,
    479: 534,
    480: 535,
    481: 536,
    482: 537,
    483: 538,
    484: 539,
    485: 540,
    486: 541,
    487: 542,
    487: 543,
    488: 544,
    489: 545,
    490: 546,
    491: 547,
    492: 548,
    492: 549,
    493: 550,
    494: 551,
    495: 552,
    496: 553,
    497: 554,
    497: 555,
    498: 556,
    499: 557,
    500: 558,
    500: 559,
    501: 560,
    502: 561,
    503: 562,
    504: 563,
    505: 564,
    506: 565,
    507: 566,
    508: 567,
    509: 568,
    510: 569,
    511: 570,
    512: 571,
    513: 572,
    514: 573,
    515: 574,
    516: 575,
    517: 576,
    518: 577,
    519: 578,
    520: 579,
    521: 580,
    522: 581,
    523: 582,
    524: 583,
    525: 584,
    526: 585,
    527: 586,
    528: 587,
    529: 588,
    530: 589,
    531: 590,
    532: 591,
    533: 592,
    533: 593,
    534: 594,
    535: 595,
    536: 596,
    537: 597,
    538: 598,
    539: 599,
    540: 600
}
REPO_DIR = op.normpath(
    git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel"))
RESULTS_DIR = op.join(REPO_DIR, 'results')
ELBOW_ASSET_DIR = op.join(REPO_DIR, 'assets', 'contactnets_elbow')

EXPERIMENT_TYPE_BY_PREFIX = {'sc': 'cube_real', 'se': 'elbow_real',
                             'va': 'vortex_asymmetric',}
                             # 've': 'vortex_elbow',
                             # 'ba': 'viscous_asymmetric',
                             # 'be': 'viscous_elbow',
                             #'gc': 'gravity_cube', 'ge': 'gravity_elbow'}
SYSTEM_BY_PREFIX = {'sc': 'cube', 'se': 'elbow', 'va': 'asymmetric'}

BAD_REAL_RUN_NUMBERS = [i for i in range(24)] + [i for i in range(25, 30)]
BAD_SIM_RUN_NUMBERS = [i for i in range(24)] + [i for i in range(25, 30)] + \
                      [31, 33, 35]
FOLDERS_TO_LOAD = [f'sweep_elbow-{i}' for i in range(2, 10)] + \
                  [f'sweep_cube-{i}' for i in range(2, 10)] + \
                  [f'sweep_cube_vortex-{i}' for i in range(2, 10)] #+ \
                  # [f'sweep_elbow_vortex-{i}' for i in range(2, 10)] + \
                  # [f'sweep_elbow_viscous-{i}' for i in range(2, 10)] + \
                  # [f'sweep_cube_viscous-{i}' for i in range(2, 10)]

N_STATE = {'cube': 13, 'elbow': 15, 'asymmetric': 13}

# RUNS_TO_LOAD = ['se30-9-0', 'se31-9-0', 'se32-9-0', 'se33-9-0', 'se34-9-0',
#                 'se35-9-0', 'se24-9-0']

PLL_TOSS_NUMS_TO_GENERATE = [0]

# If the below is true, then the generated predictions will be of the first
# trajectory of the experiment's own test set instead of a specified PLL toss
# number.  When this is the case, since the test set prediction is already
# provided in the statistics file, we will iterate over rollout horizon and
# generate many per trajectory.
DO_EXPERIMENT_TEST_SET = True

ROLLOUT_LENGTHS = [1, 2, 4, 8, 16, 32, 64, 120]


# ============================= Helper functions ============================= #
def run_name_to_run_dir(run_name):
    run_prefix = run_name[:2]
    experiment_type = EXPERIMENT_TYPE_BY_PREFIX[run_prefix]

    assert 'real' in experiment_type, \
           f'run_name: {run_name}, experiment_type: {experiment_type}'

    system = experiment_type.split('_real')[0]
    sub_number = run_name.split('-')[1]
    subfolder = f'sweep_{system}-{sub_number}'

    assert run_name in os.listdir(op.join(RESULTS_DIR, subfolder, 'runs'))

    return op.join(RESULTS_DIR, subfolder, 'runs', run_name)

def experiment_finished(run_name):
    run_dir = run_name_to_run_dir(run_name)
    return os.path.isfile(op.join(run_dir, 'statistics.pkl'))

def post_processing_done(run_name):
    run_dir = run_name_to_run_dir(run_name)
    return os.path.isfile(
        op.join(run_dir, 'post_processing', 'post_statistics.pkl'))

def load_experiment(run_name):
    run_path = run_name_to_run_dir(run_name)
    storage_name = op.abspath(op.join(run_path, '..', '..'))

    experiment_config = file_utils.load_configuration(storage_name, run_name)

    if isinstance(experiment_config.learnable_config,
                  MultibodyLearnableSystemConfig):
        experiment_config.learnable_config.randomize_initialization = False
        return DrakeMultibodyLearnableExperiment(experiment_config)
    elif isinstance(experiment_config.learnable_config,
                    DeepLearnableSystemConfig):
        return DrakeDeepLearnableExperiment(experiment_config)
    raise RuntimeError(f'Cannot recognize learnable type ' + \
                       f'{experiment_config.learnable_config}')

def get_best_system_from_experiment(exp):
    checkpoint_filename = file_utils.get_model_filename(exp.config.storage,
                                                        exp.config.run_name)
    checkpoint_dict = torch.load(checkpoint_filename)
    training_state = TrainingState(**checkpoint_dict)

    assert training_state.finished_training

    exp.learning_data_manager = ExperimentDataManager(
        exp.config.storage, exp.config.data_config,
        training_state.trajectory_set_split_indices)
    train_set, _, test_set = \
        exp.learning_data_manager.get_updated_trajectory_sets()
    learned_system = exp.get_learned_system(torch.cat(train_set.trajectories))
    learned_system.load_state_dict(training_state.best_learned_system_state)

    return learned_system

def load_ground_truth_toss_trajectory(system_name, toss_num):
    assert system_name == 'elbow'
    toss_filename = op.join(ELBOW_ASSET_DIR, f'{toss_num}.pt')
    return torch.load(toss_filename)

def compute_predicted_trajectory(
    experiment, learned_system, target_traj, system_name):
    state_n = N_STATE[system_name]
    assert target_traj.ndim == 2
    assert target_traj.shape[1] == state_n

    target_traj_list = [target_traj.reshape(1, -1, state_n)]

    predictions, targets = experiment.trajectory_predict(target_traj_list,
        learned_system, do_detach=True)

    first_state = target_traj[0].reshape(1, state_n)
    pred_traj = torch.cat((first_state,
                           predictions[0].reshape(-1, state_n)), dim=0)

    return pred_traj

def make_and_get_post_processing_dir(run_name):
    run_dir = run_name_to_run_dir(run_name)
    post_dir = op.join(run_dir, 'post_processing')
    file_utils.assure_created(post_dir)
    return post_dir

def save_predicted_bag_trajectory(predicted_traj, run_name, pll_toss_num):
    bag_toss_num = PLL_TO_BAG_NUMBERS[pll_toss_num]
    post_dir = make_and_get_post_processing_dir(run_name)
    torch.save(predicted_traj, op.join(post_dir, f'predicted_{bag_toss_num}.pt'))

def get_test_set_traj_target_and_prediction(experiment):
    stats = file_utils.load_evaluation(experiment.config.storage,
                                       experiment.config.run_name)
    test_traj_target = stats['test_model_target_sample'][0]
    test_traj_prediction = stats['test_model_prediction_sample'][0]
    
    return Tensor(test_traj_target), Tensor(test_traj_prediction)

def get_traj_with_rollout_of_len(
    traj, rollout_len, experiment, learned_system, system_name):

    state_n = N_STATE[system_name]
    assert traj.ndim == 2
    assert traj.shape[1] == state_n

    gt_traj_target = traj[(-rollout_len-1):].reshape(rollout_len+1, state_n)

    partial_pred_traj = compute_predicted_trajectory(
        experiment, learned_system, gt_traj_target, system_name)
    first_portion = traj[:(-rollout_len-1)]
    pred_traj = torch.cat((first_portion, partial_pred_traj), dim=0)

    return pred_traj

def save_rollout_sweep_trajs(rollouts, run_name):
    post_dir = make_and_get_post_processing_dir(run_name)
    for horizon, traj in zip(ROLLOUT_LENGTHS, rollouts):
        torch.save(traj, op.join(post_dir, f'test_w_horizon_{horizon}.pt'))

def compute_pos_rot_trajectory_errors(target, predicteds, experiment):
    space = experiment.space
    pos_errors, rot_errors = [], []

    # Iterate over horizon lengths.
    for tp, horizon in zip(predicteds, ROLLOUT_LENGTHS):

        running_pos_mse = 0
        running_angle_mse = 0

        for space_i in space.spaces:
            if isinstance(space_i, FloatingBaseSpace):
                pos_mse = torch.stack([space_i.base_error(tp, target)])
                angle_mse = torch.stack([space_i.quaternion_error(tp, target)])
                
                running_pos_mse += pos_mse
                running_angle_mse += angle_mse

        pos_errors.append(running_pos_mse)
        rot_errors.append(running_angle_mse)

    return pos_errors, rot_errors

def save_post_processing_stats(pos_errors, rot_errors, run_dir):
    # Make a dictionary.
    stats = {}

    for horizon, pos_error, rot_error in \
    zip(ROLLOUT_LENGTHS, pos_errors, rot_errors):
        pos_key = f'test_pos_error_w_horizon_{horizon}'
        rot_key = f'test_rot_error_w_horizon_{horizon}'

        stats[pos_key] = pos_error.item()
        stats[rot_key] = rot_error.item()

    filename = op.join(run_dir, 'post_processing', 'post_statistics.pkl')
    with open(filename, 'wb') as file:
        pickle.dump(stats, file)

def get_runs_to_load(folder, real=True):
    bad_numbers = BAD_REAL_RUN_NUMBERS if real else BAD_SIM_RUN_NUMBERS

    runs_to_load = os.listdir(op.join(RESULTS_DIR, folder, 'runs'))
    i = 0
    while i < len(runs_to_load):
        if int(runs_to_load[i][2:4]) in bad_numbers:
            runs_to_load.remove(runs_to_load[i])
        else:
            i += 1

    return runs_to_load

# ============================= Compute rollouts ============================= #
for folder in FOLDERS_TO_LOAD:
    real = True if ('viscous' not in folder and 'vortex' not in folder) \
        else False
    system = 'cube' if 'cube' in folder else 'elbow' if 'elbow' in folder else \
        'asymmetric'

    runs_to_load = get_runs_to_load(folder, real=real)

    for run_name in runs_to_load:
        if experiment_finished(run_name):
            if post_processing_done(run_name):
                print(f'{run_name} already post-processed.')
                continue
            experiment = load_experiment(run_name)
            run_dir = run_name_to_run_dir(run_name)
            print(f'Loading {run_dir}')
        else:
            print(f'Skipping unfinished {run_name}')
            continue

        learned_system = get_best_system_from_experiment(experiment)

        # If do experiment test set, use the first test set trajectory in the
        # stats file, and predict with different rollout lengths.
        if DO_EXPERIMENT_TEST_SET:
            gt_traj, pred_120 = get_test_set_traj_target_and_prediction(
                experiment)

            rollouts = []
            for rollout_len in ROLLOUT_LENGTHS[:-1]:
                rollouts.append(
                    get_traj_with_rollout_of_len(
                        gt_traj, rollout_len, experiment, learned_system,
                        system))

            rollouts.append(pred_120)

            save_rollout_sweep_trajs(rollouts, run_name)
            pos_errors, rot_errors = compute_pos_rot_trajectory_errors(
                gt_traj, rollouts, experiment)
            save_post_processing_stats(pos_errors, rot_errors, run_dir)

        # Otherwise, iterate over the PLL toss numbers.
        else:
            for pll_toss_num in PLL_TOSS_NUMS_TO_GENERATE:
                gt_traj = load_ground_truth_toss_trajectory(
                    'elbow', pll_toss_num)
                l_traj = compute_predicted_trajectory(
                    experiment, learned_system, gt_traj, system)

                save_predicted_bag_trajectory(l_traj, run_name, pll_toss_num)

pdb.set_trace()
# ======================= Compute metrics on rollouts ======================== #


