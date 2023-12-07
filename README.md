# aircraft-maintanence-classifiers

# opening note
This is a personal repository to store the code I contributed to this project and therefore does not contain the entire thing. The data used to train the models is not included for privacy reasons.

# Notre Dame/Boeing Natural Language Processing for Aircraft Maintenance Project
This project was created for Notre Dame's EG 35101 Industry and Community-Based Innovation Projects in collaboration with Boeing Co. The goal of the project was to create a tool that can take in a description of performed aircraft maintanence and output the correct log information for the maintenance log.

Contributers: Dagny Brand (dbrand@nd.edu), Jaylen Choi (jchoi22@nd.edu), Vinny Galassi (vgalassi@nd.edu), Gabby Klee (gklee@nd.edu), Richard Montoya (rmontoya@nd.edu), Joseph Park (jpark29@nd.edu), Tram Trinh (htrinh@nd.edu), Anthony Tsiantis (atsianti@nd.edu), Kevin Xue (kxue2@nd.edu), and Swindar Zhou (kzhou3@nd.edu)

# Working with our Classifiers

This project uses a Scalable Linear Support Vector Machine (SVC) classifier from the [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) library. The classifier is originally laid out in [classifer_colab](classifier_colab.ipynb). The data used to train the model was provided by Boeing. 

To use text as input to a classifier, the text must first be embedded, which changes the text data into a numerical vector representation. We use [Google's Universal Sentence Encoder](https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/tensorFlow2/variations/universal-sentence-encoder/versions/2?tfhub-redirect=true) to embed the text input. The model is downloaded and used in the "Text Embedding" section of the Colab notebook. The "SVC Classifier - one column input" section of the Colab contains our classifier model. input_data is set to the desired input column (we use descrep_narrative) and output_data is set to the desired output column (for the model classifier, this is the "wuc" column, which is column C of the data set). The train_test_split() function from sklearn splits the given input and output data into 70% training data used to train the model and 30% testing data used to test the model's accuracy. We then declare our SVC model using sklearn.svm.svc() and use the .fit() method with our training data to train the classifier. We test the accuracy using the .score() method on the testing data. The "SVC Classifier - all input columns" uses all three given input columns as input data.
