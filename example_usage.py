from run_pipeline import MLStartPipeline
pipeline = MLStartPipeline("auto_mpg.csv","mpg")
pipeline.evaluate_and_recommend_model()