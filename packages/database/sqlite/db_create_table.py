from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import *
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()


def base():
    return Base


class Analysis(Base):
    """
    Main table created for each analysis.
    db_filename: associated database file
    dataset: associated dataset
    settings: dictionary containing all analysis settings
    """

    __tablename__ = 'analysis'
    id = Column(Integer, primary_key=True)
    datetime = Column(DateTime, default=datetime.datetime.utcnow)
    db_filename = Column(String)
    dataset = Column(String)
    settings = Column(JSON)
    elapsed_time = Column(Float)
    deleted = Column(Boolean)

    """ One-to-Many """
    neural_networks = relationship("NeuralNetwork", back_populates="analysis")
    hyperparameters = relationship("HyperParameter", back_populates="analysis")
    leakage_models = relationship("LeakageModel", back_populates="analysis")
    datasets = relationship("Dataset", back_populates="analysis")
    callbacks = relationship("Callback", back_populates="analysis")
    metrics = relationship("Metric", back_populates="analysis")
    sca_metrics = relationship("ScaMetric", back_populates="analysis")
    visualizations = relationship("Visualization", back_populates="analysis")
    confusion_matrices = relationship("ConfusionMatrix", back_populates="analysis")
    random_states = relationship("RandomStates", back_populates="analysis")

    def __repr__(self):
        return "<Analysis(datetime=%s, script='%s')>" % (self.datetime, self.db_filename)


class NeuralNetwork(Base):
    __tablename__ = "neural_network"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    seed = Column(Integer)
    analysis_id = Column(Integer, ForeignKey('analysis.id'))

    """ Many-to-One """
    analysis = relationship("Analysis", back_populates="neural_networks")

    """ One-to-Many """
    layers = relationship("Layer", back_populates="neural_network")

    def __repr__(self):
        return "<NeuralNetwork(id=%d)>" % self.id


class Layer(Base):
    __tablename__ = "layer"

    id = Column(Integer, primary_key=True)
    index = Column(Integer)
    name = Column(String)
    units = Column(JSON)
    layer_type_id = Column(Integer, ForeignKey('layer_type.id'))
    activation_id = Column(Integer, ForeignKey('activation.id'))
    weight_initializer_id = Column(Integer, ForeignKey('weight_initializer.id'))
    bias_initializer_id = Column(Integer, ForeignKey('bias_initializer.id'))
    neural_network_id = Column(Integer, ForeignKey('neural_network.id'))

    """ Many-to-One """
    neural_network = relationship("NeuralNetwork", back_populates="layers")
    layer_type = relationship("LayerType", back_populates="layers")
    activation = relationship("Activation", back_populates="layers")
    weight_initializer = relationship("WeightInitializer", back_populates="layers")
    bias_initializer = relationship("BiasInitializer", back_populates="layers")

    def __repr__(self):
        return "<Layer(id=%d)>" % self.id


class LayerType(Base):
    __tablename__ = "layer_type"

    id = Column(Integer, primary_key=True)
    type = Column(String)

    """ One-to-Many """
    layers = relationship("Layer", back_populates="layer_type")

    def __repr__(self):
        return "<LayerType(id=%d)>" % self.id


class Activation(Base):
    __tablename__ = "activation"

    id = Column(Integer, primary_key=True)
    activation = Column(String)

    """ One-to-Many """
    layers = relationship("Layer", back_populates="activation")

    def __repr__(self):
        return "<Activation(id=%d)>" % self.id


class WeightInitializer(Base):
    __tablename__ = "weight_initializer"

    id = Column(Integer, primary_key=True)
    weight_initializer = Column(String)

    """ One-to-Many """
    layers = relationship("Layer", back_populates="weight_initializer")

    def __repr__(self):
        return "<WeightInitializer(id=%d)>" % self.id


class BiasInitializer(Base):
    __tablename__ = "bias_initializer"

    id = Column(Integer, primary_key=True)
    bias_initializer = Column(String)

    """ One-to-Many """
    layers = relationship("Layer", back_populates="bias_initializer")

    def __repr__(self):
        return "<BiasInitializer(id=%d)>" % self.id


class HyperParameter(Base):
    __tablename__ = 'hyperparameter'

    id = Column(Integer, primary_key=True)
    hyperparameters = Column(JSON)
    analysis_id = Column(Integer, ForeignKey('analysis.id'))

    """ Many-to-One """
    analysis = relationship("Analysis", back_populates="hyperparameters")

    """ Many-to-Many (Association)"""
    metrics = relationship("HyperParameterMetric", back_populates="hyperparameter")
    sca_metrics = relationship("HyperParameterScaMetric", back_populates="hyperparameter")

    def __repr__(self):
        return "<HyperParameter(id=%d)>" % self.id


class LeakageModel(Base):
    __tablename__ = 'leakage_model'

    id = Column(Integer, primary_key=True)
    leakage_model = Column(JSON)
    analysis_id = Column(Integer, ForeignKey('analysis.id'))

    """ Many-to-One """
    analysis = relationship("Analysis", back_populates="leakage_models")

    def __repr__(self):
        return "<LeakageModel(id=%d)>" % self.id


class Dataset(Base):
    __tablename__ = 'dataset'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    filepath = Column(String)
    analysis_id = Column(Integer, ForeignKey('analysis.id'))

    """ Many-to-One """
    analysis = relationship("Analysis", back_populates="datasets")

    def __repr__(self):
        return "<Dataset(id=%d)>" % self.id


class Callback(Base):
    __tablename__ = 'callback'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    analysis_id = Column(Integer, ForeignKey('analysis.id'))

    """ Many-to-One """
    analysis = relationship("Analysis", back_populates="callbacks")

    def __repr__(self):
        return "<Callback(id=%d)>" % self.id


class Metric(Base):
    __tablename__ = 'metric'

    id = Column(Integer, primary_key=True)
    values = Column(JSON)
    label = Column(String)
    analysis_id = Column(Integer, ForeignKey('analysis.id'))

    """ Many-to-One """
    analysis = relationship("Analysis", back_populates="metrics")

    """ Many-to-Many (Association)"""
    hyperparameters = relationship("HyperParameterMetric", back_populates="metric")

    def __repr__(self):
        return "<Metric(id=%d)>" % self.id


class ScaMetric(Base):
    __tablename__ = 'sca_metric'

    id = Column(Integer, primary_key=True)
    guessing_entropy = Column(JSON)
    success_rate = Column(JSON)
    report_interval = Column(Integer)
    label = Column(String)
    analysis_id = Column(Integer, ForeignKey('analysis.id'))

    """ Many-to-One """
    analysis = relationship("Analysis", back_populates="sca_metrics")

    """ Many-to-Many (Association)"""
    hyperparameters = relationship("HyperParameterScaMetric", back_populates="sca_metric")

    def __repr__(self):
        return "<ScaMetric(id=%d)>" % self.id


class Visualization(Base):
    """
    This table contains visualization results for a specific epoch during training.
    Visualization methods:
    - Input gradients
    - Layer-wise activation path (https://eprint.iacr.org/2019/722.pdf)
    - Layer-wise relevance propagation (LRP)
    """

    __tablename__ = 'visualization'

    id = Column(Integer, primary_key=True)
    values = Column(JSON)
    epoch = Column(Integer)
    label = Column(String)
    hyperparameters_id = Column(Integer)
    analysis_id = Column(Integer, ForeignKey('analysis.id'))

    """ Many-to-One """
    analysis = relationship("Analysis", back_populates="visualizations")

    def __repr__(self):
        return "<Visualization(id=%d)>" % self.id


class ConfusionMatrix(Base):
    __tablename__ = 'confusion_matrix'

    id = Column(Integer, primary_key=True)
    y_pred = Column(JSON)
    y_true = Column(Integer)
    hyperparameters_id = Column(Integer)
    analysis_id = Column(Integer, ForeignKey('analysis.id'))

    """ Many-to-One """
    analysis = relationship("Analysis", back_populates="confusion_matrices")

    def __repr__(self):
        return "<ConfusionMatrix(id=%d)>" % self.id


class RandomStates(Base):
    __tablename__ = 'random_states'

    id = Column(Integer, primary_key=True)
    random_states = Column(JSON)
    label = Column(String)
    index = Column(Integer)
    analysis_id = Column(Integer, ForeignKey('analysis.id'))

    """ Many-to-One """
    analysis = relationship("Analysis", back_populates="random_states")

    def __repr__(self):
        return "<RandomStates(id=%d)>" % self.id


class HyperParameterMetric(Base):
    """ Association table (hyperparameter_id, metric_id)"""

    __tablename__ = 'hyperparameter_metric'
    hyperparameter_id = Column(Integer, ForeignKey('hyperparameter.id'), primary_key=True)
    metric_id = Column(Integer, ForeignKey('metric.id'), primary_key=True)

    """ Many-to-Many (Association)"""
    metric = relationship("Metric", back_populates="hyperparameters")
    hyperparameter = relationship("HyperParameter", back_populates="metrics")

    def __repr__(self):
        return f"<HyperParameterMetric(hyperparameter_id={self.hyperparameter_id}, metric_id={self.metric_id})>"


class HyperParameterScaMetric(Base):
    """ Association table (hyperparameter_id, sca_metric_id)"""

    __tablename__ = 'hyperparameter_sca_metric'
    hyperparameter_id = Column(Integer, ForeignKey('hyperparameter.id'), primary_key=True)
    sca_metric_id = Column(Integer, ForeignKey('sca_metric.id'), primary_key=True)

    """ Many-to-Many (Association)"""
    sca_metric = relationship("ScaMetric", back_populates="hyperparameters")
    hyperparameter = relationship("HyperParameter", back_populates="sca_metrics")

    def __repr__(self):
        return f"<HyperParameterScaMetric(hyperparameter_id={self.hyperparameter_id}, sca_metric_id={self.sca_metric_id})>"
