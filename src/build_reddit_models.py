'''
Created on Mar 5, 2022

@author: paepcke
'''
from collections import namedtuple
import os

from process_data import DataPrepper
from train import EmpathyTrainer


class RedditModelBuilder:
    '''
    classdocs
    '''

    #------------------------------------
    # __init__
    #-------------------


    def __init__(self):
        '''
        Constructor
        '''
        self.cur_dir  = os.path.dirname(__file__)
        
        data_dir = os.path.abspath(os.path.join(self.cur_dir, '../dataset'))
        self.save_model_dir = os.path.abspath(os.path.join(self.cur_dir, '../reddit_models'))
        if not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)

        self.model_build_parms = namedtuple('ModelParms',
                                            ['lr',
                                            'lambda_EI',
                                            'lambda_RE',
                                            'dropout',
                                            'max_len',
                                            'batch_size',
                                            'epochs',
                                            'seed_val',
                                            'do_validation'
                                            'do_test'
                                            ])
        self.model_build_parms.lr            = 2e-5 
        self.model_build_parms.lambda_EI     = 0.5  
        self.model_build_parms.lambda_RE     = 0.5  
        self.model_build_parms.dropout       = 0.1  
        self.model_build_parms.max_len       = 64   
        self.model_build_parms.batch_size    = 32   
        self.model_build_parms.epochs        = 4    
        self.model_build_parms.seed_val      = 12   
        self.model_build_parms.do_validation = False
        self.model_build_parms.do_test       = False

        # No testing or validation:
        self.model_build_parms.test_path  = None
        self.model_build_parms.dev_path   = None

        # Build emotional reaction model:
        ER_reddit_model = self.build_emotion_reaction_model(data_dir)
        EX_reddit_model = self.build_emotion_exploration_model(data_dir)
        IN_reddit_model = self.build_interpretation_model(data_dir)

        print(f"Models are in {self.save_model_dir}")
        
    #------------------------------------
    # build_emotion_reaction_model
    #-------------------
    
    def build_emotion_reaction_model(self, data_dir):
        
        # Process raw emotion reaction reddit file
        # into data usable for training:
        infile  = os.path.join(data_dir, 'emotional-reactions-reddit.csv')
        outfile = os.path.join(data_dir, 'emotional-reactions-reddit_trainset.csv')
        save_model_path = os.path.join(self.save_model_dir, 'emotional_reactions_reddit_model.pth')
             
        print(f"Processing {infile} into a training set...")
        DataPrepper(infile, outfile)
        print(f"Done processing {infile} into a training set.")

        # Augment the common args to training:        
        self.model_build_parms.train_path = outfile
        self.model_build_parms.save_model_path = save_model_path 
        
        print(f"Training emotion reaction model...")
        trainer = EmpathyTrainer(self.model_build_parms)
        return trainer.model
    
    #------------------------------------
    # build_emotion_exploration_model
    #-------------------
    
    def build_emotion_exploration_model(self, data_dir):
        
        # Process raw emotion reaction reddit file
        # into data usable for training:
        infile  = os.path.join(data_dir, 'explorations-reddit.csv')
        outfile = os.path.join(data_dir, 'explorations-reddit_trainset.csv')
        save_model_path = os.path.join(self.save_model_dir, 'explorations_reddit_model.pth')
             
        print(f"Processing {infile} into a training set...")
        DataPrepper(infile, outfile)
        print(f"Done processing {infile} into a training set.")

        # Augment the common args to training:        
        self.model_build_parms.train_path = outfile
        self.model_build_parms.save_model_path = save_model_path 
        
        print(f"Training exlorations model...")
        trainer = EmpathyTrainer(self.model_build_parms)
        return trainer.model

    #------------------------------------
    # build_interpretation_model
    #-------------------
    
    def build_interpretation_model(self, data_dir):
        
        # Process raw emotion reaction reddit file
        # into data usable for training:
        infile  = os.path.join(data_dir, 'interpretations-reddit.csv')
        outfile = os.path.join(data_dir, 'interpretations-reddit_trainset.csv')
        save_model_path = os.path.join(self.save_model_dir, 'interpretations_reddit_model.pth')
             
        print(f"Processing {infile} into a training set...")
        DataPrepper(infile, outfile)
        print(f"Done processing {infile} into a training set.")

        # Augment the common args to training:        
        self.model_build_parms.train_path = outfile
        self.model_build_parms.save_model_path = save_model_path 
        
        print(f"Training interpretations model...")
        trainer = EmpathyTrainer(self.model_build_parms)
        return trainer.model

# ------------------------ Main ------------
if __name__ == '__main__':
    
    RedditModelBuilder()