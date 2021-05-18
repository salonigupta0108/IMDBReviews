import streamlit as st

st.title('Comparison of different algorithms for Sentiment Analysis using IMDB Reviews Dataset')
option = st.selectbox('Choose an option to read more',('Overview','Preprocessing', 'Logistic Regression', 'Multinomial Naive Bayes', 'Random Forest', 'Decision Tree', 'Vader','Conclusion'))
if option == 'Logistic Regression':
    st.markdown("Logistic regression is one of the most popular Machine Learning algorithms, which comes under the Supervised Learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables. Regression uses a more complex cost function, this cost function can be defined as the ‘Sigmoid function’ or also known as the ‘logistic function’ instead of a linear function. Logistic regression predicts the output of a categorical dependent variable. Therefore the outcome must be a categorical or discrete value. It can be either Yes or No, 0 or 1, true or False, etc. but instead of giving the exact value as 0 and 1, it gives the probabilistic values which lie between 0 and 1. Linear Regression is used for solving Regression problems, whereas Logistic regression is used for solving the classification problems. In Logistic regression, instead of fitting a regression line, we fit an 'S' shaped logistic function, which predicts two maximum values (0 or 1).")
    st.markdown("For performing Logistic Regression, I first divided the dataset into training and testing in the ratio of 8:2. Then using sklearn I imported the Logistic Regression model and trained the training set with value of C (Inverse of regularization strength) as 2 and max_iters (maximum iterations) as 5000.")
    st.markdown("The accuracy percentage of this model was found to be 89.98%.")
    st.markdown('The confusion matrix and classifictaion report are found to be as follows:')
    
    st.image('Confusion Matrices\Logistic Regression.png', caption="Confusion Matrix", width=400)
    st.markdown('')
    st.image('Classification Reports\Logistic Regression.png', caption="Classification Report", width=400)
    
elif option == 'Multinomial Naive Bayes':
    st.markdown('Multinomial Naive Bayes algorithm is a probabilistic learning method that is mostly used in Natural Language Processing (NLP). The algorithm is based on the Bayes theorem and predicts the tag of a text such as a piece of email or newspaper article. It calculates the probability of each tag for a given sample and then gives the tag with the highest probability as output.')
    st.markdown('Naive Bayes Classifier Algorithm are probabilistic algorithms based on applying Bayes’ theorem with the "naive" assumption of conditional independence between every pair of a feature. Bayes theorem calculates probability P(c|x) where c is the class of the possible outcomes and x is the given instance which has to be classified, representing some certain features.')
    st.markdown('P(c|x) = P(x|c) * P(c) / P(x)')
    st.markdown('For implementing Multinomial Naive Bayes, I have used the MultinomialNB() model from sklearn package. I first divided the dataset into training and testing in the ratio of 8:2 and then trained the model.')
    st.markdown("The accuracy percentage of this model was found to be 88.26%.")
    st.markdown('The confusion matrix and classifictaion report are found to be as follows:')
    st.image('Confusion Matrices\Multinomial NB.png', caption="Confusion Matrix", width=400)
    st.markdown('')
    st.image('Classification Reports\Multinomial NB.png', caption="Classification Report", width=400)
                  
    
elif option == 'Random Forest':
    st.markdown('Random forest is a supervised learning algorithm. The "forest" it builds, is an ensemble of decision trees, usually trained with the "bagging" method. The general idea of the bagging method is that a combination of learning models increases the overall result. Random forest builds multiple decision trees and merges them together to get a more accurate and stable prediction.')
    st.markdown('Random forests is considered as a highly accurate and robust method because of the number of decision trees participating in the process. It does not suffer from the overfitting problem. The main reason is that it takes the average of all the predictions, which cancels out the biases. Random forests is slow in generating predictions because it has multiple decision trees. Whenever it makes a prediction, all the trees in the forest have to make a prediction for the same given input and then perform voting on it. This whole process is time-consuming.')
    st.markdown('For implementing Random Forest Classifier, I have used the RandomForestClassifier() model from sklearn package. I first divided the dataset into training and testing in the ratio of 8:2 and then trained the model. The max_depth i.e. the maximum depth of the tree, was set to 4. If None, then nodes are expanded until all leaves are pure.')
    st.markdown("The accuracy percentage of this model was found to be 82.65%.")
    st.markdown('The confusion matrix and classifictaion report are found to be as follows:')
    st.image('Confusion Matrices\Random Forest.png', caption="Confusion Matrix", width=400)
    st.markdown('')
    st.image('Classification Reports\Random Forest.png', caption="Classification Report", width=400)
    
elif option == 'Decision Tree':
    st.markdown('A decision tree is a flowchart-like tree structure where an internal node represents feature (or attribute), the branch represents a decision rule, and each leaf node represents the outcome. The topmost node in a decision tree is known as the root node. It learns to partition on the basis of the attribute value. In a decision tree, each leaf node is assigned a class label. The non-terminal nodes, which include the root and other internal nodes, contain attribute test conditions to separate records that have different characteristics.')
    st.markdown('Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation. It is simple to understand and to interpret. It requires little data preparation. It is able to handle multi-output problems. However, decision trees can be unstable because small variations in the data might result in a completely different tree being generated. It can create over-complex trees that do not generalise the data well.')
    st.markdown('For implementing Decision Tree Classifier, I have used the DecisionTreeClassifier() model from sklearn package. I first divided the dataset into training and testing in the ratio of 8:2 and then trained the model. The max_depth i.e. the maximum depth of the tree was set to 3. If None, then nodes are expanded until all leaves are pure.')
    st.markdown("The accuracy percentage of this model was found to be 82.65%.")
    st.markdown('The confusion matrix and classifictaion report are found to be as follows:')
    st.image('Confusion Matrices\Decision Tree.png', caption="Confusion Matrix", width=400)
    st.markdown('')
    st.image('Classification Reports\Decision Tree.png', caption="Classification Report", width=400)
    
elif option == 'Overview':
    st.markdown("This project is a comparative study of different algorithms to implement Sentiment Analysis. The data for analysis is the 'IMDB Reviews Dataset' which has been taken from Kaggle. The dataset has 50,000 reviews of movies of different genres provided by users. It has 50% negative and 50% positive reviews.")
    st.markdown("I have used 5 different algorithms for Sentiment Analysis. 4 of them are based on training and testing while the fifth one uses a built-in 'vader-lexicon' library in python. The first step for Sentiment Analysis is to preprocess the data.")
    st.markdown("(Please refer to the 'Preprocessing' option in the drop-down.)")
    
elif option == 'Preprocessing':
    st.markdown("For preprocessing the data, I have used the following steps:")
    st.markdown("Step 1: Split all the words in each review and convert into lowercase.")
    st.markdown("Step 2: Remove all stopwords. Stopwords are words in a language that do not add meaning to the sentence and can be removed without changing meaning of the sentence.")
    st.markdown("Step 3: Remove all punctuations from the dataset.")
    st.markdown("Step 4: Lemmatization: Return the roots all the words in the reviews.")
    st.markdown("The difference between stemming and lemmatization is that stemming may form root words which do not make sense i.e. it might not return an actual word whereas lemmatizations will always return a meaningful word which is a part of the language.")
    st.markdown("Step 5: Vectorization: Using CountVectorizer to convert all reviews into binary form (1s and 0s) for proper training of data")
elif option=='Conclusion':
    st.markdown("From the observations made, it can be said that the Logisitic Regression model has the highest accuracy followed by Multinomial Naive Bayes.")
    st.markdown("The accuracies of both Random Forest Classifier and Decision Tree Classifier was found to be same i.e. 82.65% since both of them uses the concept of trees to classify and represent data.")
    st.markdown("The accuracy percentage of vader was found out be 84.016% which was better than Random Forest and Decision Tree classifiers but could not beat Logistic Regression and Multinomial Naive Bayes models.")
else:
    st.markdown("VADER (Valence Aware Dictionary and Sentiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. It is used for sentiment analysis of text which has both the polarities i.e. positive/negative. VADER is used to quantify how much of positive or negative emotion the text has and also the intensity of emotion. It also has a compound score which has the aggregate score of negative and positive combined. If the compound score of any review is greater than 0, then the review is classified as positive, else, it is classified as negative.")
    st.markdown("It does not require any training data. It can very well understand the sentiment of a text containing emoticons, slangs, conjunctions, capital words, punctuations etc. i.e. it does not require preprocessing.")
    st.markdown("I used nltk to download the 'vader_lexicon' and then imported and initialised the model by using the SentimentIntensityAnalyzer() function.")
    st.markdown("The percentage of positive reviews after vader was found to be 65.984% and the percentage of negative reviews was found to be 34.016% in contrast to the original 50% both positive and negative reviews.")
    st.markdown("Hence, it can be concluded that the percentage of error in classification of reviews using vader is 15.984%.") 
    