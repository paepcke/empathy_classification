'''
Created on Mar 21, 2022

@author: paepcke
'''

import logging
import os
import tempfile
import time
import webbrowser

import matplotlib
import torch
from transformers import RobertaTokenizer
from trulens import visualizations as viz
from trulens.nn.attribution import IntegratedGradients
from trulens.nn.models import get_model_wrapper
from trulens.nn.slices import Cut, OutputCut
from trulens.utils import tru_logger

from empathy_classifier import EmpathyClassifier
import pandas as pd


# Empathy:
class TruLensTester:
    '''
    Demonstrates how this project's model can be examined
    with TruLens.
    '''

    def __init__(self):
        '''
        Detect cuda support, and run model
        over test input.
        '''

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.logger = logging.getLogger(tru_logger.__name__)

        cur_dir = os.path.dirname(__file__)
        self.empathy_proj_root = os.path.abspath(os.path.join(cur_dir, '..'))
        self.renderer, self.output = self.verify_empathy()

    #------------------------------------
    # verify_empathy
    #-------------------

    def verify_empathy(self):
        '''
        Creates the project's three models:
        empathy emotional reaction, interpretation, 
        and exploration. Then demonstrates TruLens
        analysis on the emotional reaction classifier. 
        The others are analyzed analogously.
        
        The model is inserted into a TruLens wrapper, and
        run over an example input pair (<seeker post>, <response post>).
        The resulting per-word attributions are displayed.
        '''

        model_path = os.path.join(self.empathy_proj_root, 'reddit_models')
        ER_model_path = os.path.join(model_path, 'emotional_reactions_reddit_model.pth')
        IP_model_path = os.path.join(model_path, 'interpretations_reddit_model.pth')
        EX_model_path = os.path.join(model_path, 'explorations_reddit_model.pth')
        
        input_path = os.path.join(self.empathy_proj_root, 
                                  'dataset/sample_test_input_from_paper.csv')

        self.logger.log(logging.INFO,"Creating models...")
        empathy_classifier = EmpathyClassifier(
            self.device,
            ER_model_path = ER_model_path, 
            IP_model_path = IP_model_path,
            EX_model_path = EX_model_path,)
        self.logger.log(logging.INFO,"Done creating models.")

        
        # The numpy genfromtxt() and loadtxt() misbehave
        # with sample input data; so use pandas to get df
        #     ['id', 'seeker_post', 'response_post']
        input_df = pd.read_csv(input_path, header=0)
        
        # We don't need the ID column:
        inputs = input_df[['seeker_post', 'response_post']].to_numpy()

        # The model outputs both, a sentence's overall
        # scores on the three classes (no, some, lots),
        # and for each word whether or not it was used 
        # as a rationale in decision making. The latter
        # information is lost when we wrap the model 
        # with trulens and process input through that wrapper.
        # If desired, uncomment the following, and get:

        # ER_original_output = self.original_empathy_classification(empathy_classifier, inputs)
        
        # Now introduce TruLens:

        self.logger.log(logging.INFO,"Creating TruLens wrappers")
        # Wrap each model in a TruLensEmpathyWrapper
        ER_task = TruLensEmpathyWrapper(empathy_classifier.model_ER,
                                        RobertaTokenizer.from_pretrained('roberta-base', 
                                                                         do_lower_case=True)
                                        )
        IP_task = TruLensEmpathyWrapper(empathy_classifier.model_IP,
                                        RobertaTokenizer.from_pretrained('roberta-base', 
                                                                         do_lower_case=True)
                                        )
        EX_task = TruLensEmpathyWrapper(empathy_classifier.model_EX,
                                        RobertaTokenizer.from_pretrained('roberta-base', 
                                                                         do_lower_case=True)
                                        )
                                        
        ER_task.wrapper = get_model_wrapper(ER_task.model, 
                                            input_shape=(None, ER_task.tokenizer.model_max_length), 
                                            device=ER_task.device,
                                            backend='pytorch'
                                            )

        IP_task.wrapper = get_model_wrapper(IP_task.model, 
                                            input_shape=(None, IP_task.tokenizer.model_max_length), 
                                            device=IP_task.device,
                                            backend='pytorch'
                                            )

        EX_task.wrapper = get_model_wrapper(EX_task.model, 
                                            input_shape=(None, EX_task.tokenizer.model_max_length), 
                                            device=EX_task.device,
                                            backend='pytorch'
                                            )
        # There will be two encoders in this model:
        # the seeker_encoder, and the responder_encoder.
        # You examine them separately. First, the responder_encoder.
        # Follow the same steps for the seeker_encoder by
        # replacing 'responder_encoder' in the Cut() below:
        self.logger.log(logging.INFO,"Computing integrated gradients") 
        infl_max_ER = IntegratedGradients(
            model = ER_task.wrapper,
            doi_cut=Cut('responder_encoder_roberta_embeddings_word_embeddings'),
            qoi_cut=OutputCut(accessor=lambda o: o)
        )
        
        # As an example: examine the first seeker/response pair only.
        # From the tokenizer we will get:
        # 
        # {'input_ids': 	 tensor([[    0,  2387,  1086,  ... ],
        #               	         [    0,   100,  1346,  ... ]]), 
        #  'attention_mask': tensor([[1, 1, ...],
        #                            [1, 1,...]])
        # }
        # The two input_ids tensor rows are the seeker_post and
        # response_post, respectively:

        self.logger.log(logging.INFO,"Tokenizing...")
        tokens, attention_masks = ER_task.tokenizer(list(inputs[0]), 
                                                    padding=True, 
                                                    return_tensors='pt').values()

        # For sanity checking, uncomment the following print
        # statement to see how the tokernizer worked. 
        # Normally, decode would give us a single string for each sentence but we would
        # not be able to see some of the non-word tokens there. Flattening first gives
        # us a string for each input_id.                                                    
        # print(ER_task.tokenizer.batch_decode(torch.flatten(tokens)))

        # Move everything the GPU or CPU:
        tokens = tokens.to(ER_task.device)
        attention_masks = attention_masks.to(ER_task.device)

        # The empathy forward() model expects four arguments:
        #    seeker_tokens
        #    responder_tokens
        #    seeker_attention_mask
        #    responder_attention_mask
        #
        # For simplicity, take just the first posting/response pair from
        # the input examples, though the code could be modified
        # to do all the subsequent evaluation in a loop
        # throught the inputs
        
        model_args = [tokens[0].unsqueeze(0), 
                      tokens[1].unsqueeze(0), 
                      attention_masks[0].unsqueeze(0),
                      attention_masks[1].unsqueeze(0)]
        
        # Run the model through the wrapper, getting the attributions from
        # TruLens:
        self.logger.log(logging.INFO,"Computing attributions...")
        attrs = infl_max_ER.attributions(model_args)
        
        # attrs.shape: (1, <input_width>, 768), where
        # input_width is 22 in this example.
        
        # Collect results as tuples (<word>, <attribution-magnitude>):
        self.logger.log(logging.INFO,"Constructing per-word attribution sentence(s)")
        word_attributions = []
        for token_ids, token_attr in zip(tokens.unsqueeze(0), attrs):
            # Tokens are: torch.Size([2, 22]), where the 2
            # is the pairs seeker_post and response_post.
            # We are interested in the empathy of the response
            # only:
            response_tokens = token_ids[1,:]
            for token_id, token_attr in zip(response_tokens, token_attr):
                # Note that each `word_attr` has a magnitude for each of the embedding
                # dimensions, of which there are many. We aggregate them for easier
                # interpretation and display.
                attr = token_attr.sum()
        
                word = ER_task.tokenizer.decode(token_id)
                word_attributions.append((word, attr))
        
                # print(f"{word}({attr:0.3f})\n", end=' ')
        
        # Have TruLens generate an output of the sentences
        # with attribution strengths. How the strengths are
        # displayed depends on whether this code is running
        # in a terminal or in Jupyer/Colab. When possible, 
        # attribution strength will be shown as text colors:
        out_frags = []
        self.render_platform = self.isnotebook()
        # self.logger.log(logging.INFO,f"Running in a {self.render_platform}")
        
        try:
            renderer = viz.IPython()
        except ImportError:
            # TruLens cannot find IPython
            renderer = viz.PlainText()
        
        #self.logger.log(logging.INFO,f"Renderer is of type {type(renderer)}")
        # For each word, have TruLens create a string
        # snippet for the final output:
        for word, magnitude in word_attributions:
            out_frags.append(renderer.magnitude_colored(word, magnitude))
        #**********
        # Use alternative color scheme
        words     = [word_info[0] for word_info in word_attributions]
        attr_mags = [word_info[1] for word_info in word_attributions]

        # Pick a colormap; see https://matplotlib.org/3.5.1/tutorials/colors/colormaps.html:
        cmap = matplotlib.cm.get_cmap('BuGn')
        
        # Map logits to [0,1]:
        normed_mags = matplotlib.colors.Normalize(vmin=min(attr_mags), 
                                                  vmax=max(attr_mags))(attr_mags)
        
        # Pick colors from the colormap, proportional to logit value:
        colors = []
        for mag in normed_mags:
            # Pick color from colormap, dropping the
            # (always 255) opacity value:
            colors.append(cmap(mag, bytes=True)[:3])
            
        # Get the whole sentence as HTML, escaping occurrences
        # of '<':
        output = ' '.join([f"<span title='{mag:0.3f}' style='margin: 1px; padding: 1px; border-radius: 4px; background: black; color: rgb{color};'>{word.replace('<', '&lt')}</span>"
                           for word, color, mag
                           in zip(words, colors, attr_mags)])
        if self.render_platform == 'terminal':
            self.render_to_web(output)
        #**********        
        
        return (renderer, output)

    #------------------------------------
    # original_empathy_classification
    #-------------------
    
    def original_empathy_classification(self, empathy_classifier, inputs):
        
        for seek_post, resp_post in inputs:
            seek_tokens, seek_attention_masks = \
                empathy_classifier.tokenizer(seek_post,
                                             padding=True, 
                                             return_tensors='pt').values()
            resp_tokens, resp_attention_masks = \
                empathy_classifier.tokenizer(resp_post,
                                             padding=True, 
                                             return_tensors='pt').values()

            # Normally decode would give us a single string for each sentence but we would
            # not be able to see some of the non-word tokens there. Flattening first gives
            # us a string for each input_id.                                                    
            resp_words = empathy_classifier.tokenizer.batch_decode(torch.flatten(resp_tokens))
                                                        
            seek_tokens          = seek_tokens.to(empathy_classifier.device)
            seek_attention_masks = seek_attention_masks.to(empathy_classifier.device)
            resp_tokens          = resp_tokens.to(empathy_classifier.device)
            resp_attention_masks = resp_attention_masks.to(empathy_classifier.device)
        
            # The empathy forward() model expects four arguments:
            #    seeker_tokens
            #    responder_tokens
            #    seeker_attention_mask
            #    responder_attention_mask

            model_args = [seek_tokens,
                          resp_tokens,
                          seek_attention_masks,
                          resp_attention_masks]
        
            # For each classifier, get the following structure:
            #    ([logit_low, logit_medium, logit_high],
            #     [[logit_not_a_rationale, logit_is_a rationale],
            #      [logit_not_a_rationale, logit_is_a rationale],
            #           ...
            #     ])
            # where the first tuple element is the empathy judgement
            # of the response overall. The second tuple element contains
            # a 2-element array for each response token. Both elements
            # are logits. The first is for "this token used as part of
            # rationale" the second is "this token not used as part of
            # rationale":
            
            ER_ratings, ER_rationale_logits = empathy_classifier.model_ER(*model_args)
            IP_ratings, IP_rationale_logits = empathy_classifier.model_IP(*model_args)
            EX_ratings, EX_rationale_logits = empathy_classifier.model_EX(*model_args)

            # Bundle up the results. The 'xx_rationales' are
            # dicts mapping response text words to 1 or 0, depending
            # on whether or not the word was used as a rationale:
            resp_dict = {
                'ER_overall_logits' : ER_ratings,
                'ER_rationales'     : {word : is_rationale
                                       for word, is_rationale
                                       in zip(resp_words, torch.argmax(ER_rationale_logits, 2))},
                'IP_overall_logits' : IP_ratings,
                'IP_rationales'     : {word : is_rationale
                                       for word, is_rationale
                                       in zip(resp_words, torch.argmax(IP_rationale_logits, 2))},
                'EX_overall_logits' : EX_ratings,
                'EX_rationales'     : {word : is_rationale
                                       for word, is_rationale
                                       in zip(resp_words, torch.argmax(EX_rationale_logits, 2))}
                
                }
            return resp_dict

# ------------ Utilities ----------

    #------------------------------------
    # setup_file_paths
    #-------------------
    
    def setup_file_paths(self):
        cur_dir = os.path.dirname(__file__)
        workspace_path = os.path.join(cur_dir, '../../..')
        self.empathy_proj_root = os.path.join(cur_dir, 
                                              f"{workspace_path}/empathy_classification")

    #------------------------------------
    # isnotebook
    #-------------------

    def isnotebook(self):
        '''
        Tests whether current process is running in a:
              o terminal as a regular Python shell
              o jupyter notebook
              o Google colab
        returns one of {'terminal', 'jupyter', 'colab', None}
        None means could not determine.
        '''
        try:
            from IPython import get_ipython
        except ImportError:
            return 'terminal'
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return 'jupyter'   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return 'terminal'  # Terminal running IPython
            elif shell == 'Shell' and  get_ipython().__class__.__module__ == 'google.colab._shell':
                return 'colab'
            else:
                return 'terminal'  # Other type (?)
        except NameError:
            return 'terminal'      # Probably standard Python interpreter

    #------------------------------------
    # render_to_web
    #-------------------
    
    def render_to_web(self, output):
        '''
        Given an HTML string, open the default browser, and
        display the string there.
        
        The name of a temp file is returned. It will be 
    
        :param output: HTML to display
        :type output: str
        :return temp file path
        :rtype str
        '''
        fd = tempfile.NamedTemporaryFile(prefix='attrs_', suffix='.html')
        fd.write(bytes('<html>', 'utf8'))
        fd.write(bytes(output, 'utf8'))
        fd.write(bytes('</html>', 'utf8'))
        fd.flush()
        self.logger.info("Opening page in browser...")
        webbrowser.open_new_tab(f"file://{fd.name}")
        # Terrible hack! Must ensure the browser has read
        # the file before removing the temp file. The right
        # way would be to check for presence of some page
        # element:
        time.sleep(5)
        return fd.name
    

# -------------TruLens Wrapper Models -----------------

class TruLensEmpathyWrapper:
    MODEL = f"empathy_classifier"

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    labels = ['no', 'weak', 'strong']

    NEGATIVE = labels.index('no')
    NEUTRAL = labels.index('weak')
    POSITIVE = labels.index('strong')

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

# ------------------------ Main ------------
if __name__ == '__main__':
    tester = TruLensTester()
    tester.renderer.render(tester.output)


