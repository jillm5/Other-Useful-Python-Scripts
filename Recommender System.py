
# coding: utf-8

# # Recommender System

# In[1]:


import pandas as pd
import numpy as np


# In[41]:


rate=pd.read_csv(r'D:\data_science\Recommender\Ex_Files_Intro_Python_Rec_Systems\Exercise Files\01_02\rating_final.csv')
chef=pd.read_csv(r'D:\data_science\Recommender\Ex_Files_Intro_Python_Rec_Systems\Exercise Files\01_02\chefmozcuisine.csv')
geodata=pd.read_csv(r'D:\data_science\Recommender\Ex_Files_Intro_Python_Rec_Systems\Exercise Files\01_03\geoplaces2.csv',encoding = 'mbcs')


# In[43]:


print(rate.describe())
print(chef.describe())
print(geodata.describe())


# In[44]:


print(list(rate.columns))
print(list(chef.columns))
print(list(geodata.columns))


# In[12]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


rate.head()


# In[16]:


chef.head()


# Popularity based recommender systems

# In[21]:


#place rating
rate_group=rate.groupby('placeID', sort=False)[['rating','food_rating','service_rating']].sum().reset_index()
rate_group.sort_values(by=['rating'], ascending=False)


# In[22]:


plt.scatter(rate_group['rating'], rate_group['food_rating'])


# In[27]:


import seaborn as sns; sns.set(color_codes=True)

g=sns.lmplot(x='rating', y='food_rating', data=rate_group)


# In[28]:


rate_count=pd.DataFrame(rate.groupby('placeID')['rating'].count())
rate_count.sort_values('rating', ascending=False)


# In[33]:


most=pd.DataFrame([135085, 132825, 135032, 135052], index=np.arange(4), columns=['placeID'])
summ=pd.merge(most, chef, on='placeID')
summ


# In[35]:


chef['Rcuisine'].describe()


# ## Correlation-Based Recommendation System
# * Use pearson's r correlation to reocmmend an item that is most similiar to the item a user has already chosen
# * Item-Base similarity: how correlated are two items based on user ratings?
# * Pearson correlation coefficient (r between 1 and -1)
# * basic form of collaborative filtering

# In[45]:


rate.head()


# In[46]:


geodata.head()


# In[48]:


places=geodata[['placeID', 'name']]
places.head()


# In[49]:


chef.head()


# ## Grouping and Ranking Data

# In[51]:


ratings=pd.DataFrame(rate.groupby('placeID')['rating'].mean())
ratings.head()


# In[52]:


ratings['rating_count']=pd.DataFrame(rate.groupby('placeID')['rating'].count())
ratings.head()


# In[53]:


ratings.describe()


# In[55]:


ratings.sort_values('rating_count', ascending=False).head()


# In[57]:


places[places['placeID']==135085]


# In[59]:


chef[chef['placeID']==135085]


# ## Preparing Data for Analysis

# In[61]:


places_cross=pd.pivot_table(data=rate, values='rating', index='userID',
                              columns='placeID')
places_cross.head()


# In[63]:


toras_rate=places_cross[135085]
toras_rate[toras_rate>=0]


# ## Evaluate Similarity Based on Correlation

# In[69]:


sim_toras=places_cross.corrwith(toras_rate)
df_sim_toras=pd.DataFrame(sim_toras, columns=['PearsonR'])
df_sim_toras.dropna(inplace=True)
df_sim_toras.reset_index( inplace=True)
df_sim_toras.head()


# In[74]:


#join toras data with the main rate data
toras_corr_summ=df_sim_toras.join(ratings['rating_count'], on='placeID')
toras_corr_summ.head()


# In[76]:


toras_corr_10=toras_corr_summ[toras_corr_summ['rating_count']>=10].sort_values('PearsonR', ascending=False).head()
toras_corr_10.head()


# In[80]:


#take out reviews given by one person across more than 1 place
places_corr_toras=pd.DataFrame([135085, 132754, 135045, 135062, 135028, 135042, 135046], index=np.arange(7), columns=['placeID'])
summ=pd.merge(places_corr_toras, chef, on ='placeID')
summ


# In[81]:


places[places['placeID']==135046]


# In[82]:


chef['Rcuisine'].describe()


# ## Machine Learning for Recommendar System Build
# * classification-based collaborative filtering
#     * Naive Bayes Classification
#     * Logistic Regression 
#    => a simple ML method to predict the value of a numeric categorical       variable based on its relationship with predictor variables

# In[90]:


from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression


# In[88]:


bank=pd.read_csv(r'D:\data_science\Recommender\Ex_Files_Intro_Python_Rec_Systems\Exercise Files\02_01\bank_full_w_dummy_vars.csv')
mtcars=pd.read_csv(r'D:\data_science\Recommender\Ex_Files_Intro_Python_Rec_Systems\Exercise Files\02_03\mtcars.csv')
list(bank.columns)


# In[91]:


bank.head()


# In[92]:


bank.info()


# In[94]:


X = bank.ix[:,(18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36)].values

y = bank.ix[:,17].values


# In[99]:


LogReg=LogisticRegression()
LogReg.fit(X,y)


# In[109]:


new_user = bank.ix[1:,(18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36)].values
new_user
#[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
y_pred = LogReg.predict(new_user)
y_pred


# ## Model based collaborative filtering systems
# * SVD Matrix Factorization

# In[111]:


import sklearn
from sklearn.decomposition import TruncatedSVD


# In[112]:


#https://grouplens.org/datasets/movielens/100k/
columns = ['user_id', 'item_id', 'rating', 'timestamp']
frame = pd.read_csv(r'D:\data_science\Recommender\Ex_Files_Intro_Python_Rec_Systems\Exercise Files\02_02\ml-100k\u.data', sep='\t', names=columns)
frame.head()


# In[113]:


columns = ['item_id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
          'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies = pd.read_csv(r'D:\data_science\Recommender\Ex_Files_Intro_Python_Rec_Systems\Exercise Files\02_02\ml-100k\u.item', sep='|', names=columns, encoding='latin-1')
movie_names = movies[['item_id', 'movie title']]
movie_names.head()


# In[114]:


combined=pd.merge(frame, movies, on='item_id')
combined.head()


# In[116]:


combined.groupby('item_id')['rating'].count().sort_values(ascending=False).head()


# In[120]:


filter=combined['item_id']==50
combined[filter]['movie title'].unique()


# In[128]:


#create utility matrix
rate_cross=combined.pivot_table(values='rating',  index='user_id', columns='movie title',fill_value=0)
rate_cross.shape


# In[129]:


X=rate_cross.values.T
X.shape


# ## Decompose the matrix
# 

# In[131]:


SVD=TruncatedSVD(n_components=12, random_state=17)
result_matrix=SVD.fit_transform(X)
result_matrix.shape


# ## Generate a Correlation Matrix

# In[132]:


corr_mat=np.corrcoef(result_matrix)
corr_mat.shape


# ## Isolating a movie from the corr matrix

# In[137]:


movie_name=rate_cross.columns
movie_list=list(movie_name)
star_wars=movie_list.index('Star Wars (1977)')
print(star_wars)


# In[139]:


corr_star=corr_mat[star_wars]
corr_star.shape


# ## Recomending a highly Correlated Movie

# In[142]:


list(movie_name[(corr_star<1.0) & (corr_star>0.9)])


# In[143]:


list(movie_name[(corr_star<1.0) & (corr_star>0.95)])


# ## Content based Recommender
# * Recommends an item based on its features and how similar they are to features of other tiesm in the data set
# * Nearest neighbor algorithm - unsupervised classifier or memory-based system
# * It memorizes instances and then recommends an item (a single instance) based on how quantitatively similar it is to a new, incoming instance 
# 

# In[145]:


import sklearn
from sklearn.neighbors import NearestNeighbors


# In[146]:


mtcars=pd.read_csv(r'D:\data_science\Recommender\Ex_Files_Intro_Python_Rec_Systems\Exercise Files\02_03\mtcars.csv')
mtcars.columns = ['car_names', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
mtcars.head()


# In[148]:


t = [15, 300, 160, 3.2]

X = mtcars.ix[:,(1, 3, 4, 6)].values
X[0:5]


# In[149]:


nn=NearestNeighbors(n_neighbors=1).fit(X)


# In[150]:


print(nn.kneighbors([t]))


# In[151]:


mtcars


# ## Model Evaluation
# * How relevant the recommendation
#     * Precision- number of items recommended and matched/number of items recommended 
# * How completely did the recommender system predict the items that matched
#     * Recall-number of items recommended and matched/number of items matched  

# In[152]:


from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# In[153]:


bank.head()


# In[154]:


bank.info()


# In[156]:


X = bank.ix[:,(18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36)].values
y = bank.ix[:,17].values


# In[157]:


LogReg = LogisticRegression()
LogReg.fit(X, y)
y_pred = LogReg.predict(X)


# In[158]:


print(classification_report(y, y_pred))

