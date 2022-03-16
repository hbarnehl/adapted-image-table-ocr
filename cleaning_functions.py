import pandas as pd
import numpy as np
import re
import glob
import os

# Defining Folders
folder = '/home/hennes/Internship/constituencies/'
save_folder = '/home/hennes/Internship/constituencies_edit/'
old = '/home/hennes/Internship/old_files/'
candidates = pd.read_excel('/home/hennes/Internship/Party_Data_2021.xlsx')
PC_AC = set(sorted([folder.split('-')[0]+'-'+ folder.split('-')[1] for folder in next(os.walk(old))[1]]))
PC_AC_dict = {e.split('-')[1]: e.split('-')[0] for e in PC_AC}


def cleaning(constituency, candidate_df, year, A_serial = None, max_digits = 4, max_value = None):
    ''' This function performs the cleaning of scanned AC files.
    
    Options:
    
    year                This is relevant for the combination with the candidate & party table, which 
                        has a slightly different format every year.
    
    A-serial            Should be turned on for tables that contain one serial number with the letter
                        "A" in them. It will ignore the steps required to handle this column and will
                        carry out alternative steps where necessary.
                        
    max_digits          This specifies the maximum number of digits cells are allowed to have before
                        they are turned into NAN. This is helpful because parliamentary elections have
                        constituencies with more people, usually max 1500. These will be turned to NAN
                        with standard options.
                        
    max_value           This specifies the maximum values cells are allowed to have before they get
                        turned to NAN. This is helpful to delete noise and the summary rows.''' 
    
    df = pd.read_csv(folder+constituency)

    ################# Preliminary Cleaning #################
    
    # make F nan
    repl_dict = {'F': np.NaN} 
    df = df.replace(repl_dict, regex=True)

    # delete columns that have only have a few values in them. They are most likely useless.
    df.dropna(thresh=len(df) - (len(df)/2), axis=1, inplace=True)
    # delete rows that have more than 5 missing values
    df.dropna(thresh = (len(df.columns)/1.4), axis = 0, inplace = True)
    df.reset_index(drop=True, inplace=True)

    # transform systematic errors
    repl_dict = {'\$':'5',
                 'S':'5',
                '\(4\)':'(A)',
                '4\)':'(A)',
                '(\(A\))|(A\))|(\(A)|A':'A',
                '(\.0)$':'',
                'v':'0',
                '_':'',
                '\]':'',
                '\[':'',
                '\|':'',
                '\.':'',
                '[\(\)]':'',
                ' ': '',
                '(?!A)\D':''} 
    df = df.replace(repl_dict, regex=True)
    
    # This is needed to properly replace empty strings with NAN
    df = df.replace(r'\s+( +\.)|#',np.nan,regex=True).replace('',np.nan)

    # replace values with max_digits or more digits with NAN
    regex_str = '\d{' + str(max_digits) + ',}'
    repl_dict = {regex_str: np.NaN} 
    df = df.replace(repl_dict, regex=True)

    # delete rows that have more than 3 missing values
    df.dropna(thresh = (len(df.columns)-3), axis = 0, inplace = True)
    df.reset_index(drop=True, inplace=True)

    ############## Give Meaningful Column Names #################
    
    # The two columns with the highest numbers should be total valid votes and total votes.
    # Total valid votes is to the left of total votes.
    # first need to convert columns to int.
    
    if A_serial:
        # In case of year with A-numbers, should only do that with the non-serial number columns.
        # This mask selects all columns that do not have 'A' in them.
        mask = df[[e for e in df.columns]].apply(lambda x:
                                                 x.astype(str).str.contains(r'A', regex=True)).any(axis='index')

        # Define two masks, one with all columns except serial number, second one only serial number
        serial = [df.iloc[:,2].name]
        not_serial = df.loc[:,df.columns != serial[0]].columns.tolist()

        # convert all remaining characters to numeric or nan
        for col in not_serial:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].astype(float)
            if max_value:
                df[df[col] > max_value] = np.nan
                df.reset_index(drop=True, inplace=True)

    else:
        # convert all remaining characters to numeric or nan
        for col in df.columns.tolist():
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].astype(float)
    
        if max_value:
            df[df > max_value] = np.nan
            df.reset_index(drop=True, inplace=True)

    # name column with highest median value 'total votes'    
    df.rename(columns = {df.median().idxmax(axis=1):'total'}, inplace=True)

    # exclude 'total votes' and name the one with second highest median 'total_valid'
    columns = [col for col in df.columns if not col.startswith('total')]
    df.rename(columns = {df[columns].median().idxmax(axis=1):'total_valid'}, inplace=True)
    
    # sometimes total and total valid will be switched.
    # Check if column three places to left of total_valid is called total
    # If so, switch their names
    if df.columns.tolist()[df.columns.tolist().index('total_valid')-3] == 'total':
        df.rename(columns = {'total': 'total_valid', 'total_valid':'total'}, inplace=True)

    # get names of columns one and two places to right of 'total valid'
    try:
        sublist = ['total_valid']
        rejected = df.columns.tolist()[(df.columns.get_indexer(sublist)+1)[0]]
        nota = df.columns.tolist()[(df.columns.get_indexer(sublist)+2)[0]]
        first = df.columns.tolist()[0]
        second = df.columns.tolist()[1]
        third = df.columns.tolist()[2]
        
        # in 2019, sometimes there is only one serial column. To find out when that is the case,
        # I calculate the median euclidian distance between columns 2 and 3. If it is smaller than
        # 30, then these two are probably both serial numbers.
        if not A_serial:
            distance = []
            for row in df.index.tolist():
                distance.append(abs(df.loc[row, second] - df.loc[row, third]))

            row_dist = np.nanmedian(distance)
            
            if row_dist < 31:
                # rename first column 'page_idx' and other columns
                df.rename(columns={rejected:'rejected',
                               nota:'nota',
                               first:'page_idx',
                               second:'serial_1',
                               third:'serial'}, inplace = True)
            elif row_dist >= 31:
                # rename first column 'page_idx' and other columns
                df.rename(columns={rejected:'rejected',
                               nota:'nota',
                               first:'page_idx',
                               second:'serial'}, inplace = True)
                
        else:
            df.rename(columns={rejected:'rejected',
               nota:'nota',
               first:'page_idx',
               second:'serial_1',
               third:'serial'}, inplace = True)
            
    except Exception as e:
        print(e)
        print('Something went wrong with the column naming')
        return

        # rename first column 'page_idx' and other columns
        df.rename(columns={rejected:'rejected',
                           nota:'nota',
                           first:'page_idx',
                           second:'serial_1',
                           third:'serial'}, inplace = True)


    # delete all rows in which no cell has more than two digits (also accounting for .0)

    mask = df.apply(lambda x: x.astype(str).str.contains(r'^\d{,2}(\.0)?$', regex=True)).all(axis=1)
    df = df[~mask]
    # sometimes that does not work, so to be sure: keep only rows with sum of >300
    df = df.loc[df.sum(1) >= 350]
    df.reset_index(drop=True, inplace=True)

    ############## correcting columns wrongly shifted #######################

    # Identify all rows that have NAN in the column furthest right
    
    if A_serial:
        # and which do not have an 'A' in the serial column. Those should not be moved
        rowlist = df[(df.iloc[:,-1].isna()) & (~df['serial'].str.contains('A', na=False))].index.tolist()
    else:
        rowlist = df[(df.iloc[:,-1].isna())].index.tolist()

    # Calculate how many standard deviations all values of each row are away from the average of the respective columns 
    for row in rowlist:
        collist = df.dtypes[df.dtypes == float].index.tolist()
        sdlist_old = []
        for col in collist:
            sdlist_old.append(abs(df.loc[row, col] - df[col].mean()) / df[col].std()) ## abs returns positive numbers

    # Compute the average standard deviation for each of these rows 
        rowsd_old = np.nanmedian(sdlist_old)

    # Shift the values of the row to the right and report the new average standard deviation 
        df1 = df.copy(deep=True)
        df1.loc[row, :] = df1.loc[row, :].shift(1, axis=0)
        collist = df1.dtypes[df1.dtypes == float].index.tolist()
        sdlist_new = []
        for col in collist:
            sdlist_new.append(abs(df1.loc[row, col] - df[col].mean()) / df1[col].std())

        rowsd_new = np.nanmedian(sdlist_new)

    # Take over the shift if the new SD is smaller than the old SD

        if rowsd_old > rowsd_new:
            df.loc[row] = df1.loc[row]
            
            if A_serial:
                # make sure that all except serial are still float
                serial = [df.iloc[:,2].name]
                not_serial = df.loc[:,df.columns != serial[0]].columns.tolist()

                # then convert all remaining characters to numeric or nan
                for col in not_serial:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].astype(float)

    ####################### Correct Serial Numbers #############################

    # For errors without a gap (no row is missing)

    # Idea is: if (n)+1 does not equal (n+1), then see if (n-1)+2 equals (n+1)
    # This logic is extended for up to 5 numbers ahead of n. In this way, gaps 
    # of up to 4 numbers will be bridged. At the same time, there will be no
    # interpolation if there is no clean continuation of integers.

    for n in df.index.tolist()[1:]:
        try:
            if df.iloc[n,1] != df.iloc[n-1,1]+1:
                if df.iloc[n-1,1]+2 == df.iloc[n+1,1]:
                    df.iloc[n,1] = df.iloc[n-1,1]+1
                if df.iloc[n-1,1]+3 == df.iloc[n+2,1]:
                    df.iloc[n,1] = df.iloc[n-1,1]+1
                    df.iloc[n+1,1] = df.iloc[n-1,1]+2
                if df.iloc[n-1,1]+4 == df.iloc[n+3,1]:
                    df.iloc[n,1] = df.iloc[n-1,1]+1
                    df.iloc[n+1,1] = df.iloc[n-1,1]+2
                    df.iloc[n+2,1] = df.iloc[n-1,1]+3
                if df.iloc[n-1,1]+5 == df.iloc[n+4,1]:
                    df.iloc[n,1] = df.iloc[n-1,1]+1
                    df.iloc[n+1,1] = df.iloc[n-1,1]+2
                    df.iloc[n+2,1] = df.iloc[n-1,1]+3
                    df.iloc[n+3,1] = df.iloc[n-1,1]+4
                if df.iloc[n-1,1]+6 == df.iloc[n+5,1]:
                    df.iloc[n,1] = df.iloc[n-1,1]+1
                    df.iloc[n+1,1] = df.iloc[n-1,1]+2
                    df.iloc[n+2,1] = df.iloc[n-1,1]+3
                    df.iloc[n+3,1] = df.iloc[n-1,1]+4
                    df.iloc[n+4,1] = df.iloc[n-1,1]+5
        except:
            None

    # For errors with a gap (a row is missing)
    # define index of last row of each page
    last = [x-1 for x in df[df['page_idx']==0].index.tolist()[1:]]

    # if (n)+1 does not equal (n+1), then see if (n-1)+2 equals (n+1)
    for n in last:
        try:
            if df.iloc[n,1] != df.iloc[n-1,1]+1:
                df.iloc[n,1] = df.iloc[n-1,1]+1
        except:
            None
    
    # for errors at beginning of pages

    for n in df[df['page_idx']==0].index.tolist()[1:]:
        try:
            if df.iloc[n,1] != df.iloc[n+1,1]-1:
                df.iloc[n,1] = df.iloc[n+1,1]-1
        except:
            None
            
    ########################## Correct Serial Numbers with A ##############################
    
    if A_serial: # only do this if there is indeed a column with A
        # fill nan with empty strings
        df.iloc[:,2].fillna('', inplace = True)
        # set cells with only an "A" to "0A"
        mask = df.loc[:,"serial"]=='A'
        df.loc[mask, "serial"] = "0"
        # set cells with empty string to "0"
        mask = df.loc[:,"serial"]==''
        df.loc[mask, "serial"] = "0"
        # for all rows except last row
        for n in df.index.tolist()[1:-1]:
            
            # preceding and following row should not contain any A
            if not ("A" in "".join([str(df.iloc[n-1,2]), str(df.iloc[n+1,2])]) and not
                # n should not be identical to preceding or following row
                int(''.join(c for c in str(df.iloc[n,2]) if c.isdigit())) ==
                int(''.join(c for c in str(df.iloc[n+1,2]) if c.isdigit())) and not
                int(''.join(c for c in str(df.iloc[n,2]) if c.isdigit())) ==
                int(''.join(c for c in str(df.iloc[n-1,2]) if c.isdigit()))):
                # if n-1 == (n+1)-1, then make n = n-1+A
                if (int(''.join(c for c in str(df.iloc[n-1,2]) if c.isdigit())) ==
                    int(''.join(c for c in str(df.iloc[n+1,2]) if c.isdigit()))-1):
                    df.iloc[n,2] = ''.join([str(int(''.join(c for c in str(df.iloc[n-1,2]) if c.isdigit()))), 'A'])
                # in cases where n-1 = n+1 -2 (then n should not have an A and simply be n-1 + 1)
                elif (int(''.join(c for c in str(df.iloc[n-1,2]) if c.isdigit())) ==
                      int(''.join(c for c in str(df.iloc[n+1,2]) if c.isdigit()))-2):
                    df.iloc[n,2] = int(''.join(c for c in str(df.iloc[n-1,2]) if c.isdigit()))+1
            
            # if the former row ends with an A and does not start with an A
            if str(df.iloc[n-1,2]).endswith("A") and not str(df.iloc[n-1,2]).startswith("A"):
                # then just make n = former row + 1
                df.iloc[n,2] = int(''.join(c for c in df.iloc[n-1,2] if c.isdigit()))+1
            # if the next row ends with an A and does not start with an A
            elif str(df.iloc[n+1,2]).endswith("A") and not str(df.iloc[n+1,2]).startswith("A"):
                # then just make n = next row
                df.iloc[n,2] = int(''.join(c for c in df.iloc[n+1,2] if c.isdigit()))
            

       

    ################# Rename Candidate Columns According to Parties #######################
    if year == 2021:
        # get appropriate constituency number
        con_n = float(re.sub(r'AC0*', '' ,constituency.split('.')[0]))

        # define df excluding NOTA, only current constituency
        dat = candidate_df[(candidate_df['PARTY']!= 'NOTA')
                           & (candidate_df['AC NO.'] == con_n)][['AC NO.', 'PARTY', 'TOTAL']]
        # create column with rank of party per constituency
        dat['rank'] = dat.groupby('AC NO.').rank(ascending=False)
        # get number of candidates
        n_candidates = len(dat)
        # create dict with party value pair
        rank_party = pd.Series(dat.PARTY.values,index=dat['rank']).to_dict()

        # create dictionary with key = column name, value = rank
        serial = df.columns.get_indexer(['serial'])[0]
        column_rank = df.iloc[:,serial+1:serial+(n_candidates+1)]\
            .agg(func=np.sum)\
            .rank(ascending=False)\
            .to_dict()

        # Renaming column according to rank
        rename_dict={}
        for col, rank in column_rank.items():
            rename_dict.update({col:rank_party.get(rank)}) 
        df.rename(columns=rename_dict, inplace=True)
    
    ################# create constituency number column #######################
    
    df['ac'] = re.findall(r'[1-9][0-9]*' ,constituency)[0]
    
    return df

def connect_party(df, ACs):
    
    # get appropriate constituency number
    con_n = PC_AC_dict[ACs[0].split('.')[0]].split('C')[-1].replace('0', '')

    # define df excluding NOTA, only current constituency
    dat = candidates[(candidates['Party']!= 'NOTA') & (candidates['Constituency_No'] == int(con_n))]\
    [['Constituency_No', 'Party', 'Votes', 'Constituency_Name']]

    dat['Votes'] = pd.to_numeric(dat['Votes'], errors='coerce')
    # create column with rank of party per constituency
    dat['rank'] = dat.groupby('Constituency_No').rank(ascending=False)
    # get number of candidates
    n_candidates = len(dat)
    # create dict with party value pair
    rank_party = pd.Series(dat.Party.values,index=dat['rank']).to_dict()

    # create dictionary with key = column name, value = rank
    serial = df.columns.get_indexer(['serial'])[0]
    column_rank = df.iloc[:,serial+1:serial+(n_candidates+1)]\
        .agg(func=np.sum)\
        .rank(ascending=False)\
        .to_dict()

    df['pc'] = con_n

    # Renaming column according to rank
    rename_dict={}
    for col, rank in column_rank.items():
        rename_dict.update({col:rank_party.get(rank)}) 
    df.rename(columns=rename_dict, inplace=True)
    
    dflist = []
    for x in df['ac'].unique().tolist():
        acn = "{:03d}".format(int(x))
        df[df['ac']==x].to_csv(f'{save_folder}AC{acn}.csv')