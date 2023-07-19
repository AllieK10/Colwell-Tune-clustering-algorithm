import numpy as np
import pandas as pd
from itertools import combinations

def getExcelName():
    excel_name = input("Enter file name: ")
    return excel_name


def same_variants(lesson1, lesson2): #count how many same variants there are in two lessons
    counter = 0
    l1 = list(lesson1.values)
    l2 = list(lesson2.values)
    combination = list(zip(l1, l2))
    if users_option == "N":
        for el in combination:
            if el[0] == el[1] and el[0]!=0 and el[1]!=0:
                counter += 1
    if users_option == "Y":
        for el in combination:
            if el[0] == el[1]:
                counter += 1
    return counter


def percentage(similarities_num, variations):  #counts and rounds up the %
    number = round((similarities_num/variations)*100)
    return number


def zero_columns(lesson1, lesson2):
    counter = 0
    l1 = list(lesson1.values)
    l2 = list(lesson2.values)
    combination = list(zip(l1, l2))
    for el in combination:
        if el[0] == el[1] == 0:
            counter += 1
    return counter


def variations_count(df, lesson1, lesson2):
    if users_option == "N":
        variations_number = len(df.columns) - zero_columns(lesson1, lesson2)
    if users_option == "Y":
        variations_number = len(df.columns)  # variations = columns
    return variations_number


def transform_matrix(df):
    lessons = list(df.index.values)
    lessons_combinations = list(combinations(lessons, 2))  # AB, BC, BD, AC...
    print(lessons_combinations)
    # variations_number = len(df.columns)  # variations = columns
    # print(variations_number)
    number_of_lessons = len(df.index.values.tolist())  # lessons = rows
    # print(number_of_lessons)
    empty_matrix = np.zeros(shape=(number_of_lessons, number_of_lessons))
    # print(empty_matrix)
    symmetric_matrix_df = pd.DataFrame(empty_matrix, index=lessons, columns=lessons)  # new matrix
    # print(symmetric_matrix_df)
    for i in lessons_combinations:
        print(i)
        l1 = df.loc[i[0]]
        l2 = df.loc[i[1]]
        print(l1.values)
        print(l2.values)
        similar = same_variants(l1, l2)
        print("Number of same variants: ", similar)
        variations = variations_count(df, l1, l2)
        final_percent = percentage(similar, variations)
        print("This is % of same variants: ", final_percent)
        symmetric_matrix_df.loc[i[0], i[1]] = final_percent
        symmetric_matrix_df.loc[i[1], i[0]] = final_percent

    # print(symmetric_matrix_df)
    symmetric_matrix_df = symmetric_matrix_df.astype(int)
    # print(symmetric_matrix_df)
    df = symmetric_matrix_df
    return df


def switch_places(df, idx1, idx2):
    column_list = list(df.columns)
    # print(column_list)
    column_list[idx2], column_list[idx1] = column_list[idx1], column_list[idx2]
    new_df = df[column_list]
    row_list = df.index.values.tolist()
    # print(row_list)
    row_list[idx2], row_list[idx1] = row_list[idx1], row_list[idx2]
    new_df = new_df.reindex(row_list)
    return new_df


def is_descending(array): #function to check if array is sorted in descending order
    for i in range(1, len(array)): # array = upper or lower diagonal of matrix, we check it till the end
        if array[i-1] >= array[i]: # compare every two numbers and check if the previous one is bigger than the next one
            continue # if so, continue checking
        else:
            return False # otherwise stop, it's not descending anymore
    return True


def split_diag(array): # to get separate part of diagonal which is descending
    n = len(array)
    for i in range(n):# go from first to last number of array
        if i != n: # while it's not the last element of the array
            if is_descending(array[:i+1]) == True: # check every slice starting from the full array if it's descending
                continue
            else:#as soon as we hit an increasing number
                return array[:i] #return the part before it
    return array[:i+1] #otherwise the whole diagonal is descending and this is one cluster


user_input = input("Choose type of data to work with: symmetric matrix or asymmetric matrix. Print S or A accordingly: ")
if user_input == "A":
    df_asymm = pd.DataFrame(pd.read_excel(getExcelName(), header=None, index_col=0))
    df_asymm.index.name = None
    print(df_asymm)
    users_option = input("Count zero-columns or not? Type Y for yes or N for no: ")
    df = transform_matrix(df_asymm)
if user_input == "S":
    df_symm = pd.DataFrame(pd.read_excel(getExcelName(), index_col=0))
    df_symm.index.name = None
    # lessons = list(df_symm.index.values)
    df = df_symm

lessons = list(df.index.values) #list of row names
# print(lessons)
print("\n")
print("INITIAL MATRIX:")
print(df)

updated_df = df
basic_lesson = input("Choose a lesson: ")
if basic_lesson not in lessons:
    print("Does not exist in a list")
else:
    print("Found", basic_lesson)
    bl_idx = df.columns.get_loc(basic_lesson) #gets index of chosen basic lesson (by columns)
    # if basic_lesson is in the first column and row position, move on
    # if not, put it in the first position (switch places with the first standing lesson)
    if bl_idx == 0:
        print("it's in correct position")
        updated_df = df
        print(updated_df)  # use original dataframe for the first step
    else:
        updated_df = switch_places(df, 0, bl_idx)
        print(updated_df)


i = 0
num_rows = len(df.index)
# print(num_rows)
for i in range(i, num_rows):
    # print(i)
    df1 = updated_df.iloc[i+1:num_rows]
    if df1.empty:
        print("DataFrame is empty!")
        print("\n")
        print("THIS IS FINAL MATRIX: ")
        print(updated_df)
        break
    else:
        # print(df1)
        current_max = df1[basic_lesson].max()
        print("Current max is: ", current_max)
        max_val_index = df1[basic_lesson].idxmax()
        print("Max value is on row: ", max_val_index)
        max_idx = updated_df.columns.get_loc(max_val_index)
        updated_df = switch_places(updated_df, i+1, max_idx)
        print(updated_df)
        i+=1

print("\n")
df_new = updated_df.to_numpy() # turn a dataframe matrix into numpy array matrix

upper_diagonal = np.diag(df_new, k=1)
print("this is upper diagonal: ", upper_diagonal)
n = len(upper_diagonal)
# print(n)

result = [] #to store clusters in a list of lists
start = 0
finish = n #equal to the length of the diagonal
while start < finish: #until we hit the end
    # print("Cluster: ", split_diag(upper_diagonal[start:finish])) # cluster is a slice of a diagonal which has only descending numbers
    cluster = list(split_diag(upper_diagonal[start:finish])) #turn slice of array into a list
    # print("Type of cluster: ", type(cluster))
    slice_size = len(split_diag(upper_diagonal[start:finish])) #needed for skipping the cluster we already got to continue the loop
    # print(slice_size)
    start = start + slice_size #next starting point of this loop is the position of the first increasing
                                                    # number after the end of the previous cluster
    result.append(cluster) #add cluster we got in a loop to the list of clusters


final_clusters = []
lessons_labels = list(updated_df.columns)
num_lessons = len(lessons_labels)
pos = 0
for number, element in enumerate(result):
    print("Group №", number+1, ":", element)
    cluster_size = len(element)
    # print(cluster_size)
    # print("current pos: ", pos)
    while pos < num_lessons:
        if pos == 0:
            end = pos + cluster_size + 1
        else:
            end = pos + cluster_size
        group = lessons_labels[pos:end]
        final_clusters.append(group)
        # print(group)
        step = len(group)
        pos = pos + step
        # print("next position: ", pos)
        break
print("Final Clusters: ", final_clusters)
for num, cl in enumerate(final_clusters):
    print("Cluster №", num + 1, ":", cl)
