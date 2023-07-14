from sqlalchemy.orm import sessionmaker
from packages.database.sqlite.db_create_table import *


class DBSelect:

    def __init__(self, database_path):
        self.db_select = None
        self.engine = create_engine('sqlite:///{}'.format(database_path), echo=False)
        self.metadata = MetaData(self.engine)
        self.session = sessionmaker(bind=self.engine)()

    def select_all_analysis(self):
        return self.session.query(Analysis).all()

    def select_analysis(self, analysis_id):
        return self.session.query(Analysis).filter_by(id=analysis_id).first()

    def select_hyperparameter_from_id(self, hyperparameter_id):
        return self.session.query(HyperParameter).filter_by(id=hyperparameter_id).first()

    def select_leakage_model_from_id(self, leakage_model_id):
        return self.session.query(LeakageModel).filter_by(id=leakage_model_id).first()

    def select_dataset_from_id(self, dataset_id):
        return self.session.query(Dataset).filter_by(id=dataset_id).first()

    def select_callback_from_id(self, callback_id):
        return self.session.query(Callback).filter_by(id=callback_id).first()

    def select_metric_from_id(self, metric_id):
        return self.session.query(Metric).filter_by(id=metric_id).first()

    def select_sca_metric_from_id(self, sca_metric_id):
        return self.session.query(ScaMetric).filter_by(id=sca_metric_id).first()

    def select_visualization_from_id(self, visualization_id):
        return self.session.query(Visualization).filter_by(id=visualization_id).first()

    def select_confusion_matrix_from_id(self, confusion_matrix_id):
        return self.session.query(ConfusionMatrix).filter_by(id=confusion_matrix_id).first()

    def select_random_states_from_id(self, random_states_id):
        return self.session.query(RandomStates).filter_by(id=random_states_id).first()
