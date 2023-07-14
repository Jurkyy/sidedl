from packages.database.sqlite.db_create_table import *
from sqlalchemy.orm import sessionmaker
from sqlalchemy import and_


class DBUpdate:

    def __init__(self, database_path, analysis_id):
        self.db_update = None
        self.analysis_id = analysis_id
        self.engine = create_engine('sqlite:///{}'.format(database_path), echo=False)
        self.metadata = MetaData(self.engine)
        self.session = sessionmaker(bind=self.engine)()

    def db_update_elapsed_time(self, elapsed_time):
        """
        :param elapsed_time: elapsed time for the analysis (obtained with time.time())
        """
        self.session.query(Analysis).filter(Analysis.id == self.analysis_id).update({"elapsed_time": elapsed_time})
        self.session.commit()

    def db_update_settings(self, settings):
        """
        :param settings: dictionary containing analysis settings
        """
        self.session.query(Analysis).filter(Analysis.id == self.analysis_id).update({"settings": settings})
        self.session.commit()

    def db_update_sca_metric_label(self, old_label, new_label):
        """
        Update the label of a sca_metric row
        :param old_label: previous label
        :param new_label: new label
        """
        self.session.query(ScaMetric).filter(and_(
            ScaMetric.analysis_id == self.analysis_id,
            ScaMetric.label == old_label)).update({"label": new_label})
        self.session.commit()

    def db_update_metric_label(self, old_label, new_label):
        """
        Update the label of a metric row
        :param old_label: previous label
        :param new_label: new label
        """
        self.session.query(Metric).filter(and_(
            Metric.analysis_id == self.analysis_id,
            Metric.label == old_label)).update({"label": new_label})
        self.session.commit()

    def db_update_random_states_label(self, old_label, new_label):
        """
        Update the label of a random_states row
        :param old_label: previous label
        :param new_label: new label
        """
        self.session.query(RandomStates).filter(and_(
            RandomStates.analysis_id == self.analysis_id,
            RandomStates.label == old_label)).update({"label": new_label})
        self.session.commit()

    def db_update_hyperparameters(self, new_hyperparamaters, hyperparamaters_id):
        """
        Update hyperparameters row
        :param new_hyperparamaters: new hyperparameters
        :param hyperparamaters_id: hyperparamaters_id
        """
        self.session.query(HyperParameter).filter(
            and_(HyperParameter.id == hyperparamaters_id, HyperParameter.analysis_id == self.analysis_id)).update(
            {"hyperparameters": new_hyperparamaters})
        self.session.commit()

    def get_db_update(self):
        """
        :return: db_update object
        """
        return self.db_update

    def get_db_session(self):
        """
        :return: db session object
        """
        return self.session
