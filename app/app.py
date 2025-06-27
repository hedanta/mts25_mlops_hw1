import gzip
import os
import pandas as pd
import shutil
import time
import logging
from watchdog.observers.polling import PollingObserver 
from watchdog.events import FileSystemEventHandler
from datetime import datetime

import src.preprocessing as preproc
import src.scorer as scorer


with gzip.open('/app/models/model_catboost.cbm.gz', 'rb') as f_in:
    with open('/app/models/model_catboost.cbm', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(self):
        logger.info('Initializing ProcessingService...')
        self.input_dir = '/app/input'
        self.output_dir = '/app/output'
        self.train = preproc.load_train_data()
        logger.info('Service initialized')

    def process_single_file(self, file_path):
        try:
            logger.info('Processing file: %s', file_path)
            input_df = pd.read_csv(file_path)

            logger.info('Starting preprocessing')
            processed_df, cat_features = preproc.preprocess_data(self.train, input_df)
            
            logger.info('Making prediction')
            submission = scorer.make_pred(processed_df, cat_features, file_path)
            filename = os.path.splitext(os.path.basename(file_path))[0]
            
            logger.info('Prepraring submission file')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            preds_filename = f"predictions_{timestamp}_{os.path.basename(file_path)}"
            submission.to_csv(os.path.join(self.output_dir, preds_filename), index=False)
            logger.info('Predictions saved to: %s', preds_filename)

            json_filename = f"top_features_{timestamp}_{os.path.basename(file_path)}"
            scorer.save_feature_importance_json(self.output_dir, json_filename)
            logger.info('Top 5 feature importances saved to: %s', preds_filename)
            
            density_filename = f"preds_density_{timestamp}_{os.path.basename(file_path)}"
            scorer.save_prediction_density_plot(submission['prediction'], self.output_dir, density_filename)
            logger.info('Predictions density saved to: %s', preds_filename)

        except Exception as e:
            logger.error('Error processing file %s: %s', file_path, e, exc_info=True)
            return


class FileHandler(FileSystemEventHandler):
    def __init__(self, service):
        self.service = service
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".csv"):
            logger.debug('New file detected: %s', event.src_path)
            self.service.process_single_file(event.src_path)


if __name__ == "__main__":
    logger.info('Starting ML scoring service...')
    service = ProcessingService()
    observer = PollingObserver()
    observer.schedule(FileHandler(service), path=service.input_dir, recursive=False)
    observer.start()
    logger.info('File observer started')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
        observer.stop()
    observer.join()