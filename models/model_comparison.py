from models.model_evaluation import ModelEvaluator

class ModelComparator:
  """
  A class to compare multiple trained models and identify the best-performing one using robust evaluation.
  """

  def __init__(self, task_type, primary_metric=None):
      """
        Initialize the ModelComparator class.

        :param task_type: A string representing the type of task, either 'classification' or 'regression'.
        :param primary_metric: A string representing the primary metric for determining the best model (optional).
        """
      self.task_type = task_type
      self.primary_metric = primary_metric
      self.evaluator = ModelEvaluator(task_type)

  def compare_models(self, trained_models, X_test, y_test):
      """
        Compare multiple trained models and identify the best-performing one based on evaluation metrics.

        :param trained_models: A list of tuples where each tuple contains the model name and the trained model object.
        :param X_test: A list or ndarray representing the test features.
        :param y_test: A list or ndarray representing the test target values.
        :return: A tuple containing:
                 - best_model_name: A string indicating the name of the best-performing model.
                 - best_model: The best-performing model object.
                 - model_metrics: A dictionary containing evaluation metrics for all models.
                 - total_ranks: A dictionary containing ranking information for each model across metrics.
        """
      model_metrics = {}
      metric_ranks = {}
      total_ranks = {}

      # Evaluate models
      for name, model in trained_models:
          metrics = self.evaluator.evaluate_model(model, X_test, y_test)
          model_metrics[name] = metrics

          for metric, value in metrics.items():
              if metric not in metric_ranks:
                  metric_ranks[metric] = []
              metric_ranks[metric].append((name, value))

      # Rank models for each metric
      for metric, scores in metric_ranks.items():
          if self.task_type == "classification" or metric not in ["MAE", "MSE", "RMSE"]:  # Higher is better
              scores = sorted(scores, key=lambda x: x[1], reverse=True)
          else:  # Lower is better for regression metrics
              scores = sorted(scores, key=lambda x: x[1])

          for rank, (name, _) in enumerate(scores, start=1):
              if name not in total_ranks:
                  total_ranks[name] = {}
              total_ranks[name][metric] = rank

      # Calculate average rank
      for name in total_ranks.keys():
          ranks = list(total_ranks[name].values())
          total_ranks[name]["average_rank"] = sum(ranks) / len(ranks)

      best_model_name = min(total_ranks, key=lambda name: total_ranks[name]["average_rank"])
      best_model = next(model for model_name, model in trained_models if model_name == best_model_name)

      return best_model_name, best_model, model_metrics, total_ranks


