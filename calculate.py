import pandas as pd

number_list=[10,20,30,40,50,60,70,80,90,110,120,130,140,150]
def calculate(number):
    df = pd.read_csv("Research_Data/weebit"+str(number)+".0.test.csv")

    total_correct = sum(df['Grade'] == df['bert.2.pred'])
    total_passage = len(df['Grade'])
    print(total_correct)
    print(total_passage)
    print(total_correct/total_passage)

for number in number_list:
    print(number)
    calculate(number)
    print("end")