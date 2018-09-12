import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

def main():
	df = pd.read_csv("./winemag-data_first150k.csv")

	df = df.drop(labels=['Unnamed: 0'], axis=1)

	print(df.head())
	print(df.count())
	
	print(df.describe())
	
	for col in df.columns:
	    print("count values in column " + col)
	    print(df[col].value_counts(dropna = False))

	df[df['price'] < 100].plot.hexbin(x='price', y='points', gridsize=15)
	#the hex plot shows clustering at the 20$-86pt mark
	plt.show()

	#analasys of the description column in terminal
	#shows that there are duplicate entries
	df = df.drop_duplicates()
	print(df.count())
	#droping duplicates revealed about 30k duplicates
	#new count of dataframe is 97851

if __name__ == '__main__':
	main()