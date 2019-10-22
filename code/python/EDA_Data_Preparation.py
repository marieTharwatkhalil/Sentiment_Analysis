#!/usr/bin/env python
# coding: utf-8

# # EDA 

# In[28]:


import pandas as pd 
import numpy  as np
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[2]:


data= pd.read_csv("Tweets.csv")


# In[3]:


data.head()


# In[4]:


#important features: airline_sentiment negativereason airline text 


# In[5]:


data.shape


# In[6]:


#the data has 14640 instances and 15 features 


# In[7]:


data.info()


# In[8]:


#the important features has only 1 feature (negativereason) with Null values #"airline","retweet_count","text"


# In[9]:


#feature Selection 


# In[10]:


data_important=data.drop(labels=["tweet_id","airline_sentiment_confidence","negativereason_confidence","airline_sentiment_gold","name","negativereason_gold","tweet_coord","tweet_created","tweet_location","user_timezone"],axis=1)


# In[11]:


data_important.head()


# In[12]:


data_important.info()


# In[13]:


#the new data frame has 4 non numerical features and only 1 numerical feature 


# In[15]:


data_important.describe()


# In[16]:


# retweet_count ranges from 0 to 44 


# In[17]:


data_important[data_important["retweet_count"]>0].count()


# In[18]:


(767/14640)*100


# In[19]:


# there are 767 instances have retweet_count greater than 0 
#only 5% of the tweets were retweeted


# In[20]:


data_important["airline_sentiment"].unique()


# In[21]:


#there are 3 categories 'neutral', 'positive', 'negative'


# In[23]:


sent=data_important.groupby("airline_sentiment")["text"].nunique()
sent


# In[24]:


#there are 9087 negative tweets, 3067 neutral tweets and 2298 positive tweets 


# In[44]:


def get_percentages(sent_):
    """this function compute the percentage of each sentiment to all data """
    neg=(sent_[0]/14640)*100
    neut=(sent_[1]/14640)*100
    pos=(sent_[2]/14640)*100
    
    print(pos,"% of the tweets are Positive")
    print(neut,"% of the tweets are Neutral") 
    print(neg,"% of the tweets are Negative") 
    
    return  pos,neg, neut


# In[ ]:





# # Visualization 

# In[51]:


plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#909090'
plt.rcParams['axes.labelcolor']= '#909090'
plt.rcParams['xtick.color'] = '#909090'
plt.rcParams['ytick.color'] = '#909090'
plt.rcParams['font.size']=13


# In[262]:


def Pie_chart(sent_):
    """ a funcation that takes positive,negative,neutral values to plot a pie chart """
    
    positive,negative,neutral=get_percentages(sent_)
    labels=["Positive ["+str(round(positive,2))+"%]","Negative ["+str(round(negative,2))+"%]","Neutral ["+str(round(neutral,2))+"%]"]
    sizes=[positive,negative,neutral]
    color=["royalblue","tomato","green"]

    patches,text=plt.pie(sizes,[.1,0,0],colors=color,startangle=90,pctdistance=.6,labeldistance=.5,)
    plt.legend(patches,labels,loc="best", bbox_to_anchor=(1.5,0.8))
    plt.title("Sentiments of all tweets")
    plt.show()


# In[263]:


Pie_chart(sent)


# In[78]:


positive_total,negative_total,neutral_total=get_percentages(sent)


# In[264]:


def Pie_chart2(positive,negative,neutral,title):
    """ a funcation that takes positive,negative,neutral values to plot a pie chart """
    
    
    labels=["Positive","Negative ","Neutral"]
    sizes=[positive,negative,neutral]
    color=["royalblue","tomato","green"]
    
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.pie(sizes,[.2,0,0],colors=color,autopct='%1.0f%%',
    startangle=90, pctdistance=.8)
    ax.axis('scaled')
    ax.legend(labels,frameon=True, bbox_to_anchor=(1,0.8))
    plt.show()
    return fig 


# In[266]:


fig=Pie_chart2(positive_total,negative_total,neutral_total,"Sentiment of all tweets")


# In[267]:


fig.savefig("Sentiments of all tweets_2.png")


# In[71]:


def count_retweets_sent(data_):
    """this function takes the dataframe and count the sentiment of retweeted tweets
    then return 3 objects the total number of retweets, 
    dictionary of sentiments counts, dictionary of the occurance of each count"""
    
    rtwts_sent={}
    rtwts_sent_value_count={}
    
    for i in ["positive","negative","neutral"]:
        retweet_total=data_[data_["retweet_count"]>0]
        retweet_sent=retweet_total[retweet_total["airline_sentiment"]==i]
        len_=len(retweet_sent)
        rtwts_sent[i]=len_
        x=retweet_sent["retweet_count"].value_counts()
        rtwts_sent_value_count[i]=x.to_dict()
    
    return len(retweet_total),rtwts_sent, rtwts_sent_value_count


# In[72]:


retweet_total_count,rt_count,rt_df=count_retweets_sent(data_important)


# In[73]:


retweet_total_count


# In[74]:


rt_count


# In[75]:


rt_df


# In[76]:


def get_retweets_percentages(retweets_count,rtwt_ttl_count):
    """this function takes the dictionary that contains number of retweets for each sentiment
    and the total number of retweeted tweets then calculates the percentage of each sentiment"""
    neg=(retweets_count["negative"]/retweet_total_count)*100
    neut=(retweets_count["neutral"]/retweet_total_count)*100
    pos=(retweets_count["positive"]/retweet_total_count)*100

    
    print(pos,"% of the retweeted tweets are Positive")
    print(neut,"% of the retweeted tweets are Neutral") 
    print(neg,"% of the retweeted tweets are Negative") 
    
    return pos,neg, neut


# In[82]:


positive_retweets,negative_retweets,neutral_retweets=get_retweets_percentages(rt_count,retweet_total_count)


# In[90]:


title_2="Sentiment of retweets"


# In[268]:


fig2=Pie_chart2(positive_retweets,negative_retweets,neutral_retweets,title_2)


# fig2.savefig("Sentiment of retweets_2")

# In[93]:


data_important.head()


# In[94]:


data_important.groupby("airline")["airline_sentiment"].unique()


# In[95]:


def airlines_df(data_):
    """this function takes a dataframe 
    then creates and returns a dictionary  of all airlines 
    and the sentiments associated with each one """
    airlines={}
    for i in ["American","Delta","Southwest","US Airways","United","Virgin America"]:
        airlines[i]=data_[data_["airline"]==i]
    
    return airlines


# In[96]:


air_df=airlines_df(data_important)


# In[97]:


type(air_df.keys())


# In[98]:


def airlines_analysis(airlines_df):
    """this function takes a dataframe 
    then print the anaylsis for all airlines """
    sentments_all={}
    for i in airlines_df.keys():
        print(i)
        
        sent=airlines_df[i].groupby("airline_sentiment")["text"].nunique()
        get_percentages(sent)
        sentments_all[i]=sent.to_dict()
        rt_total,rt_sent,rt_sent_value_count=count_retweets_sent(airlines_df[i])
        get_retweets_percentages(rt_sent,rt_total)
    return sentments_all


# In[99]:


general_sentiment=airlines_analysis(air_df)


# In[100]:


general_sentiment


# In[105]:


def cout_sentiments(data_):
    """this function takes the dataframe 
    then returns the count of each sentiment and list of ailines"""
    post=[]
    negt=[]
    neutr=[]
    for i in general_sentiment.keys():

        z=general_sentiment[i]
        #print(general_sentiment[i])
        for j in z:
            #print(j,z[j])
            if j == "positive":
                post.append(z[j])
            elif j == "negative":
                negt.append(z[j])
            elif j == "neutral":
                neutr.append(z[j])
                
    return   post, negt, neutr, list(general_sentiment.keys())


# In[106]:


post, negt, neutr,labels=cout_sentiments(general_sentiment)


# In[ ]:





# In[115]:


import seaborn as sns


# In[126]:


post, negt, neutr


# In[139]:


def Bar_chart(post_,negt_,neutr_,labels_,title_):
    """this function takes sentments labels and title 
    to plot a barplot of the 3 sentiments for all airlines"""
    
    barWidth = 0.25
    bars1 = post_
    bars2 = negt_
    bars3 = neutr_

    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    plt.bar(r1, bars1, color='royalblue', width=barWidth, edgecolor='white', label='Positive')
    plt.bar(r2, bars2, color='tomato', width=barWidth, edgecolor='white', label='Negative')
    plt.bar(r3, bars3, color='green', width=barWidth, edgecolor='white', label='Neutral')

    plt.xlabel(title_, fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], labels_,rotation=20)
    
    plt.legend()
    plt.show()
    
    return plt


# In[140]:


title333='Airline Comparison'


# In[143]:


fig333=Bar_chart(post,negt,neutr,labels,title333)


# fig.savefig(title333)

# # Cleaning text

# In[147]:


text_df=data_important.drop(labels=["negativereason","airline","retweet_count"],axis=1)
text_df.head()


# In[148]:


from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
import emoji
import re


# In[149]:


def strip_emoji(text):
    """this fuction removes emojis from sentences 
    then returns cleaned sentence"""
    
    print(emoji.emoji_count(text))

    new_text = re.sub(emoji.get_emoji_regexp(), r"", text)

    return new_text 


# In[150]:


def remove_pattern(input_txt, pattern):
    """this fuction removes patterns from sentences 
    such as @makaabn or any given pattern 
    then returns cleaned sentence"""
    
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt


# In[151]:


def stop_words_remove(sent):
    """this fuction removes stop words from sentences 
    then returns cleaned sentence"""
      
    stop_words = set(stopwords.words('english')) 

    word_tokens = word_tokenize(sent) 

    filtered_sentence = [w for w in word_tokens if not w in stop_words] 

    filtered_sentence = [] 

    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 

    return filtered_sentence


# In[152]:


def clean_data(data):
    """this fuction remove unwanted chars, symbols, emojis...etc from sentences 
    then returns a cleaned dataframe"""
    
    for i in range(len(data.text)):
        print(i)
        b=str(data.text[i])
        b=b.lower()
        b=remove_pattern(b, "@[\w]*")
        b=strip_emoji(b)
        b=re.sub(r'\b\w{1,4}\b', '', b)
        b=re.sub('[^\w]', ' ',b)
        b.replace("[^a-zA-Z#]", " ")
        b.replace("http'", "")
        b=stop_words_remove(b)
        print(b)
        
        print("XXX")
        data.text[i]=b
        print(data.text[i])
        
    return data


# In[153]:


cleaned_data=clean_data(text_df)


# In[284]:


clnd_data=cleaned_data
clnd_data


# In[285]:


cleaned_data.head()
cleaned_data.to_csv(r"cleaned_tweets_2",index=False)


# In[286]:


clnd_data=pd.read_csv(r"cleaned_tweets_2")
clnd_data.shape


# In[287]:


empty=[]
for i in range(len(clnd_data)):
    
    if clnd_data.iloc[i,1]== '[]':
        #clnd_data=clnd_data.drop(i)
        empty.append(i)
        #print(i)
empty 


# In[290]:


#clnd_data.iloc[0,1]
#clnd_data=clnd_data.drop(empty)
#clnd_data.shape

clnd_data


# In[291]:


for i in range(14353):
    n=clnd_data.iloc[i,1]
    n=n.replace("[","")
    n=n.replace("]","")
    clnd_data.iloc[i,1]=n
    


# In[300]:


clnd_data.to_csv(r"cleaned_tweets_2",index=False)
#xx222=pd.read_csv("cleaned_tweets_2")
clnd_data.head()


# In[179]:


from wordcloud import WordCloud


# In[293]:


def word_cloud(data_,sent_):
    """this function takes the dataframe and the sentiment(positive,negative or neutral 
    then plot its most frequent words as a word cloud)"""
    normal_words =' '.join([text for text in data_['text'][data_['airline_sentiment'] == sent_]])

    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
    #plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title(sent_.upper()+ " most freuqent words")
    plt.show()
    
    return plt
    


# In[294]:


title_4="Positive - most frequent words"


# In[295]:


fig_4=word_cloud(clnd_data,"positive")


# #fig_4.savefig(title_4)
# #save_figure(fig_4,title_4)

# In[196]:


title_5="Negative - most frequent words"


# In[239]:


def save_figure(fig_,title_):
    
    fig_.savefig(title_)
    


# In[296]:


fig_5=word_cloud(clnd_data,"negative")


# #save_figure(fig_5,title_5)

# In[185]:


from collections import Counter 
from nltk.probability import FreqDist


# In[279]:


def words_freq_plot_sent(data_,sentiment_):
    """this function takes the dataframe and the sentiment(positive,negative or neutral 
    then plot its top 7 frequent words as a bar plot"""
    if sentiment_=="negative":
        pallette="gist_heat"
    elif sentiment_=="positive":
        pallette="Blues_r"  
    else:
        pallette="summer"   
        

    sent_text=data_[data_["airline_sentiment"]==sentiment_]
    #print(sent_text)
    #print(sentiment_)
    fdist_sent = FreqDist(sent_text["text"])
    top_ten_sent = fdist_sent.most_common(7)
    data_sent = top_ten_sent
    names_, values_ = zip(*data_sent)
    values_=list(values_)
    #print(top_ten_sent)
    
    dict_vis_={}
    #print(dict_vis_)
    x=0
    for i in names_:
        dict_vis_[i]=values_[x]
        x+=1
    #print(dict_vis_)
    sent_words_df=pd.DataFrame(dict_vis_,index=["freq"])
    sent_words_df.transpose()
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    ax2 = sns.barplot(data=sent_words_df,palette=pallette,orient='h')
    ax2.set(xlabel='Count', ylabel='most Frequent words',title=sentiment_.upper()+ " Top 7 freuqent words")
    plt.show()
    
    return fig2


# In[297]:


fig_6 = words_freq_plot_sent(clnd_data,"positive")


# In[282]:


title_6="Positive - Top 7 freuqent words"


# In[298]:


save_figure(fig_6,title_6)


# In[ ]:




