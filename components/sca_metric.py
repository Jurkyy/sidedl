from components import Component
from components.decorators.database_decorators import sca_metrics_database
from packages.sidechannel.metrics.sca_metrics import SCAMetrics


class ScaMetric(Component):

    def __init__(self, settings, model, labels_key_guess):
        super().__init__(settings)
        self.model = model
        self.labels_key_guess = labels_key_guess

    @sca_metrics_database
    def run(self, traces, nt_metrics, correct_key, db_label=""):
        sca_metrics_obj = SCAMetrics(self.model, traces, nt_metrics, self.labels_key_guess, correct_key)
        return sca_metrics_obj.run(self.settings["key_rank_executions"])
