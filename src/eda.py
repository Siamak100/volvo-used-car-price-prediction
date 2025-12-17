
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("data_insamling_volvo_blocket.xlsx")

print(df.info())
print(df.describe())

df['Försäljningspris'].hist(bins=30)
plt.title("Distribution of prices")
plt.show()
