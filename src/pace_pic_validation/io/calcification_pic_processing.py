import pandas as pd 


def pic_from_ca(df, ca_column, na_column, V_acid_column, V_filt_column, end_col_name): 

    df = df.copy()
    M_Ca = 40.08 #g/mol

    df[ca_column] = pd.to_numeric(df[ca_column])
    df[na_column] = pd.to_numeric(df[na_column])
    df[V_acid_column] = pd.to_numeric(df[V_acid_column])
    df[V_filt_column] = pd.to_numeric(df[V_filt_column])

    # seawater correction
    # mg / L 
    df[ca_column] = df[ca_column] - 0.038215379 * df[na_column]

    # mg / L -> mg in falcon
    # ca_column is in mL
    # 0.001 to multiply V filt column from mL to L

 
    m_sample = df[ca_column] * (df[V_acid_column] * 0.001) #ok

    #print(f'this is mass in sample in mg {m_sample}')

    # mg -> g 
    m_sample = m_sample * 0.001 #ok

    #print(f'this is mass in sample in g {m_sample}')

    # g -> mol through M_Ca
    M_sample = m_sample / M_Ca #ok


    # Volume Vilt in mL -> L -> m3
    df[V_filt_column] = df[V_filt_column] * 1e-6

    #print(f'this is volume in filt in m3 {df[V_filt_column]}')

    # 0.001 to multiply V filt column from mL to L
    # 0.001 to multiply V filt column from L to m3
    # 0.001 * 0.001 = 0.000001 = 1e-6

    df[end_col_name] = (M_sample / df[V_filt_column]) 

    # conversion mol to mmol

    df[end_col_name] = df[end_col_name] * 1000

    print(f'this is concentrationin mmol/m3 {df[end_col_name]}')
    
    #print('concentration given in mmol/m3')

    return df


