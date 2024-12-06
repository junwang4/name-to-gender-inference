quick: predict
#quick: evaluate advanced_error_analysis
#quick: advanced_error_analysis

predict:
	python gender_inference.py --task=predict --input=data/sample.csv

evaluate:
	python gender_inference.py --task=evaluate --input=data/PeerJ_7000.csv

advanced_error_analysis:
	python gender_inference.py --task=advanced_error_analysis --input=data/PeerJ_7000.csv
