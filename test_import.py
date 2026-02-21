import sys
sys.path.append('.')

try:
    from src.speculative_decoding import standard_speculative_decoding, static_ensemble_verification, diversed_decoding
    print('Successfully imported speculative decoding functions')
except Exception as e:
    print(f'Error importing speculative decoding functions: {e}')

try:
    from src.models import EnsembleHead, EnsembleWrapper
    print('Successfully imported model classes')
except Exception as e:
    print(f'Error importing model classes: {e}')

try:
    from train.trainer import DiversedTrainer
    print('Successfully imported DiversedTrainer class')
except Exception as e:
    print(f'Error importing DiversedTrainer: {e}')

try:
    from scripts.run_dp import main as run_dp_main
    print('Successfully imported run_dp main function')
except Exception as e:
    print(f'Error importing run_dp main function: {e}')
