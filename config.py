class Config:    
    N_ID = 2048
    UNIT_INTVL = 1/1000
    N_INTVL = 10
    
    UNIT_TIMESTEP = N_INTVL
    MAX_TIMESTEP = 300
    N_FEATURES = 30

    BATCH_SIZE = 64
    EPOCHS = 300

    MODE = 'lstm'
    isMC = True
    isTRAIN = True
    STATUS = 'D'
    N = 1

    NAME = MODE
    
    if isMC:
        NAME += '_mc'

    if isTRAIN:
        FILENAME = f"dataset/0_Preliminary/0_Training/Pre_train_{STATUS}_{N}.csv"
        DATAPATH = f"data/{NAME}/train/{STATUS}/"
    else:
        FILENAME = f"dataset/0_Preliminary/1_Submission/Pre_submit_{STATUS}.csv"
        DATAPATH = f"data/{NAME}/test/{STATUS}/"
    MODEL_NAME = f"models/{NAME}_{STATUS}.h5"
