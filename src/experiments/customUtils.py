import os
from datetime import datetime
from itertools import count
from rpyc.utils.classic import obtain

from nspyre.gui.widgets.save import save_json
from nspyre import DataSink
from nspyre import ProcessRunner

from pathlib import Path

def flexSave(datasetName:str, expType:str, filename:str, dirs:list = ['E:\\Data\\']): #TODO: Make it so this hangs up data acquisition. Will need to use ProcessRunner, multithread acq and saving, and then make them talk to each other nicely. Gross
    '''Creates a save of the data a specified directory(ies) in a Dir\\DATE(YYMMDD)\\EXPERIMENT_TYPE\\EXP_TYPE TIME(HHMMSS) SAVE_TYPE.json structure
    Arguments:  *datasetName:str, name of data to be saved from dataserv
                *expType:str, name of the experiment (or name of folder to save in under the date)
                *filename:str, file name entered in experiment GUI
                *OBSOLETE: saveType:str, typically something like auto, closeout, final
                *dirs:list, dirs ending in \\ to save data to. Default is Jasper's Data driver'''

    if not len(dirs) > 0:
        raise ValueError('No directories specified for custom autosaver')
    
    now = datetime.now()
    with DataSink(datasetName) as dataSink:
        try:
            print("Trying to save in flexSave...")
            dataSink.pop(1)
            for save_dir in dirs:
                # build platform-correct paths
                base = Path(save_dir)
                date_path = base / now.strftime('%y_%m_%d')
                exp_path = date_path / expType
                base_name = f"{expType}_{filename}"
                file_path = exp_path / f"{base_name}.json"

                print(date_path, exp_path, file_path)

                # create the full experiment directory if it doesn't already exist
                if os.path.exists(exp_path) and not os.path.isdir(exp_path):
                    raise FileExistsError(f"Cannot create directory {exp_path}: a non-directory file exists at that path.")
                exp_path.mkdir(parents=True, exist_ok=True)

                # If the file exists, append a numeric suffix (_1, _2, ...) until an unused name is found
                counter = 1
                while os.path.isfile(file_path):
                    file_path = os.path.join(exp_path, f"{base_name}_{counter}.json")
                    counter += 1

                # print("obtained data: ", obtain(dataSink.data))
                save_json(str(file_path), obtain(dataSink.data)) # save the data as .json 
        except TimeoutError:
            raise ValueError(f'No data with name \'{datasetName}\' to save (or timed out)')

def saveInNewProc(datasetName:str, expNameForAutosave:str, saveType:str, dirs:list=None):
        '''Starts a new process (that shouldn't kill current processes) to save the data to a specified directory(ies) in a Dir\\DATE(YYMMDD)\\EXPERIMENT_TYPE\\EXP_TYPE TIME(HHMMSS) SAVE_TYPE.json structure
        Arguments:  *datasetName:str, name of data to be saved from dataserv
                    *expNameForAutosave:str, str that will used in saved file
                    *saveType:str, typically something like auto, closeout, final
                    *dirs:list, dirs ending in \\ to save data to. Default is None, which uses the default save directories from flexSave in CustomUtils.py. Default is None'''
        saveProc = ProcessRunner(kill=False)
        if dirs is not None:
            saveProc.run(flexSave, datasetName, expNameForAutosave, saveType, dirs)
        else:
            saveProc.run(flexSave, datasetName, expNameForAutosave, saveType)
        return(saveProc)

def setupIters(maxIterations):
    if maxIterations < 0:
        iters = count() #infinite iterator
    else:
        iters = range(maxIterations) 
    return(iters)

def setupLaser(gw, laserPower):
    if not gw.laser.is_on():
        raise(ValueError('LASER IS NOT ON'))
    gw.laser.set_power(laserPower)

def setupSigGen(gw, freq, rfPower, sigGenName:str='vaunix'):
    sigGen = getattr(gw, sigGenName)
    sigGen.setPwrLvl(rfPower)
    sigGen.setFreq(freq)
    sigGen.setIfRFOut(True)