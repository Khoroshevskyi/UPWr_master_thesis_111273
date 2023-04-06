def zmien_nazwe(some_list: list) -> list:
    """
    zmien wszystko na polskie nazwy
    """
    new_list = []
    for k in some_list:
        new_list.append(f"Stadium {k[-1]}")
    return new_list
        

def apply_oversampling(X, y):
    X_class = X
    orig_data = pd.Series(y).value_counts().sort_index().to_dict()
    keys1 = zmien_nazwe(list(orig_data.keys()))
    values1 = list(orig_data.values())
    
    smote = SMOTE(random_state = 101)
    X_class, label_cat = smote.fit_resample(X_class, y)
    oversample_data = pd.Series(label_cat).value_counts().to_dict()
    keys2 = zmien_nazwe(list(oversample_data.keys()))
    values2 = list(oversample_data.values())

    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_under, y_under = undersample.fit_resample(X, y)
    undersample_data = pd.Series(y_under).value_counts().sort_index().to_dict()
    keys3 = zmien_nazwe(list(undersample_data.keys()))
    values3 = list(undersample_data.values())
    

    
    fig, axs = plt.subplots(1,3, figsize=(15,8))


    fig1 =  axs[0].bar(keys1, values1, color ='g', width = 0.5, )
    axs[0].set_ylabel('Ilość próbek', fontsize=15)
    axs[0].set_xlabel('Oryginalny', fontsize=15)
#     axs[0].set_title('Oryginalny', fontsize=20)

    
    fig2 = axs[1].bar(keys2, values2, color ='g', width = 0.5) 
#     axs[1].set_ylabel('Stadim', fontsize=15)
    axs[1].set_xlabel('Oversampling', fontsize=15)
#     axs[1].set_title('Oersampling', fontsize=20)
    
    axs[2].axis(ymin=0.5,ymax=700)
    fig3 = axs[2].bar(keys3, values3, color ='g', width = 0.5)  
#     axs[2].set_ylabel('Ilość próbek', fontsize=15)
    axs[2].set_xlabel('Undersampling', fontsize=15)
#     axs[2].set_title('Po undersampling', fontsize=20)
    
    #   plt.savefig("test1.svg")
    return X_class, label_cat

X_train_1, y_train_1 = apply_oversampling(X, y)
