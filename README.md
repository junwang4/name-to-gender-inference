
### A LightGBM-Based Approach to Predicting Gender Likelihood from Personal Names

## Installation

```
git clone https://github.com/junwang4/name-to-gender-inference
cd name-to-gender-inference
pip install -r requirements.txt
```

## Usage

### STEP 1. Prepare you data (for example, see file `./data/sample.csv`)
Format: `family name, first and/or middle name`
```
name
"Nawab, Amber"
"Rittinghaus, Klaus"
"Ben Moshe, Shlomit"
"Agrawal, Damyanti"
"Adkin, Allan L"
"Park, Kaechang"
```

#### STEP 2. Apply the inference model (located in: `./models`) to your data of personal names
```
python name2gender.py --task=predict --input=data/sample.csv
```

The gender prediction result will be saved to `data/sample.pred.csv`, with the following format:
```
name,male_confidence
"Nawab, Amber",0.01
"Rittinghaus, Klaus",1.00
"Ben Moshe, Shlomit",0.03
"Agrawal, Damyanti",0.16
"Adkin, Allan L",0.99
"Park, Kaechang",0.96
```

## Performance of our model

Santamaría and Mihaljević published a paper:
[Comparison and benchmark of name-to-gender inference services (2018, PeerJ)](https://peerj.com/articles/cs-156/).
They compiled a benchmark dataset of 7076 names which in turn
includes 5779 names with known gender.
They used the 5779 gender-known names 
to examine the performance of five name-to-gender services:
*Gender API*, *gender-guesser*, *genderize.io*, *NameAPI*, and *NamSor*. 

They found that typically **Gender API** achieves the best results.
For example, in Table 9 of their paper (Benchmark 3a: Minimization of misclassifications constrained to a 25% non-classification rate on the entire data set),
they reported: 
> Gender API outperforms the other services with a 0.9% misclassification rate, followed by NamSor with 1.4% and genderize.io with 1.7%.
 
Specifically, the following table shows the performance of
*Gender API* in terms of confusion matrix.

|      | M pred | F pred | U pred |
|------|--------|--------|--------|
|M     | 3,573  | 110    | 128    |
|F     | 172    | 1,750  | 46     |

In the table, the column `U pred` shows that
*Gender API* is not sure about the gender of 128 male and 46 female names (in total, 174 unknowns).

To compare with *Gender API*, 
we set the confidence thresholds for identifying Male and Female as 0.602 and 0.543, respectively.
This adjustment ensures that the 'U pred' numbers for both Male and Female 
align (though a bit different) with those generated by the *Gender API*.
(The observation that the female threshold is lower than male is due to the fact that 75% of the
training data consists of male names.)

```
python name2gender.py --task=advanced_error_analysis --input=data/PeerJ_7000.csv --confidence_M_TH=0.603 --confidence_F_TH=0.55
```

|      | M pred | F pred | U pred |
|------|--------|--------|--------|
|M     | 3,555  | 126    | 130    |
|F     | 147    | 1,777  | 44     |


The F1-score comparison between our algorithm and Gender API is as follows:

|  | Gender API | Our algorithm |
|--|------------|---------------|
|M | 0.946      | 0.946         |
|F | 0.914      | **0.918**     |


## Conclusion 

- Our model achieves performance comparable to the best among the five name-to-gender inference services.
- Furthermore, our model is open-source and offers probabilistic gender predictions.


## In practice, we suggest using the following confidence thresholds.

-	If male_confidence > 0.83, the name is predicted as male.
-	If female_confidence > 0.71, the name is predicted as female.
-	Otherwise, the name is classified as gender-unknown.

With these thresholds, the model achieves balanced recall (0.86 for both male and female) and precision (0.98 for males
and 0.97 for females).


|      | M pred | F pred | U pred |
|------|--------|--------|--------|
|M     | 3,280  | 57    | 474     |
|F     | 70    | 1,697  | 201     |

```
   precision    recall  f1-score   support
  
F      0.968     0.861     0.912      1968
M      0.979     0.862     0.916      3811
```
