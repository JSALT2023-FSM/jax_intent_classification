import functools
import flax.linen as nn
import jax
import jax.numpy as jnp
import last
import matplotlib.pyplot as plt
import optax
import tensorflow as tf
import os
import tensorflow_datasets as tfds
import numpy as np
import pickle
import orbax.checkpoint
import pandas as pd
from tqdm import tqdm
from utils import sequence_mask, flip_sequences
from utils import get_scenario_action
import dataclasses
# Disallow TensorFlow from using GPU so that it won't interfere with JAX.
tf.config.set_visible_devices([], 'GPU')
#if not jax.devices('gpu'):
#  raise RuntimeError('We recommend using a GPU to run this notebook')
@dataclasses.dataclass
class RecognitionLatticeConfig:
  encoder_embedding_size: int
  context_size: int
  vocab_size: int
  hidden_size: int
  rnn_size: int
  rnn_embedding_size: int
  locally_normalize: bool

  def build(self) -> last.RecognitionLattice:
    context = last.contexts.FullNGram(
        # In LAST's convention, blank doesn't count towards vocab_size.
        vocab_size=self.vocab_size - 1,
        context_size=self.context_size,
    )
    alignment = last.alignments.FrameDependent()

    def weight_fn_cacher_factory(context: last.contexts.FullNGram):
      return last.weight_fns.SharedRNNCacher(
          vocab_size=context.vocab_size,
          context_size=context.context_size,
          rnn_size=self.rnn_size,
          rnn_embedding_size=self.rnn_embedding_size,
      )

    def weight_fn_factory(context: last.contexts.FullNGram):
      weight_fn = last.weight_fns.JointWeightFn(
          vocab_size=context.vocab_size, hidden_size=self.hidden_size
      )
      if self.locally_normalize:
        weight_fn = last.weight_fns.LocallyNormalizedWeightFn(weight_fn)
      return weight_fn

    lattice = last.RecognitionLattice(
        context=context,
        alignment=alignment,
        weight_fn_cacher_factory=weight_fn_cacher_factory,
        weight_fn_factory=weight_fn_factory,
    )
    return lattice

train_csvs_dir  = "/home/szaiem/rich_representation_learning/data/slurp_csvs/"
encoder_dir = "/export/corpora/representations/SLURP/encoder_outputs/"

train_csv = "/home/szaiem/rich_representation_learning/data/slurp_csvs/train_real-type=direct.csv"
train_synthetic= "/home/szaiem/rich_representation_learning/data/slurp_csvs/train_synthetic-type=direct.csv"
test_csv= "/home/szaiem/rich_representation_learning/data/slurp_csvs/test-type=direct.csv"
dev_csv= "/home/szaiem/rich_representation_learning/data/slurp_csvs/devel-type=direct.csv"
def create_label_encoders(train_csvs) :
  actions_dict = {}
  scen_dict ={}
  intent_dict ={}
  intent_count = 0
  act_count = 0
  scen_count = 0
  tables = []
  for train_csv in train_csvs : 
    tables.append (pd.read_csv(train_csv))
  table = pd.concat(tables)
  semantics = list(table["semantics"])
  actions = []
  scenarios = []
  intents = []
  for sem in semantics : 
      scenario, action, intent = get_scenario_action(sem)
      intents.append(intent)
      actions.append(action)
      scenarios.append(scenario)
  unique_actions = set(actions)
  unique_scenarios = set(scenarios)
  unique_intents = set(intents)
  for act in unique_actions : 
      actions_dict[act] = act_count 
      act_count+=1
  for scen in unique_scenarios : 
      scen_dict[scen] = scen_count 
      scen_count+=1
  for intent in unique_intents : 
      intent_dict[intent] = intent_count
      intent_count +=1
      
  return scen_dict,actions_dict, scen_count, act_count, intent_dict, intent_count


scen_dict, actions_dict, scen_count,act_count, intent_dict, intent_count =create_label_encoders([train_csv, train_synthetic]) 
print(f" total number of scenarios : {scen_count}")
print(f" total number of actions : {act_count}")
print(f" total number of intents : {intent_count}")


def preprocess_example_tf(wav, scenario, action, intent):
    b= tf.numpy_function( numpy_preprocess, [wav, scenario, action, intent], [tf.float32, tf.int64, tf.int64, tf.int64, tf.int64])
    return {"encoder_frames": b[0], "action": b[2], "scenario": b[1],"intent": b[3], 'num_frames': b[4]}

def numpy_preprocess(wav, scenario, action, intent):
  encoder_frames = np.squeeze(np.load(os.path.join(encoder_dir, wav.decode("utf-8")+".npy")))
  action_label, scenario_label, intent_label = actions_dict[action.decode("utf-8")], scen_dict[scenario.decode("utf-8")], intent_dict[intent.decode("utf-8")]
  return encoder_frames,scenario_label, action_label, intent_label, encoder_frames.shape[1]

def preprocess(
    dataset: tf.data.Dataset,
    is_train: bool = True,
    batch_size: int = 4,
    max_num_frames: int =600 ,
) -> tf.data.Dataset:
  """Applies data preprocessing for training and evaluation."""
  # Preprocess individual examples.
  if is_train:
    dataset = dataset.shuffle(buffer_size=1000).repeat()
  dataset = dataset.map(preprocess_example_tf, num_parallel_calls=tf.data.AUTOTUNE)
  # Shuffle and repeat data for training.
  # Pad and batch examples.
  dataset = dataset.padded_batch(
      batch_size,
      {
      'encoder_frames' : [max_num_frames, None],
      'action' : [],
      'scenario': [],
      "intent": [],
      'num_frames': [],
      },
  )
  return dataset

def create_dict(train_csvs): 
    tables = []
    for train_csv in train_csvs : 
        tables.append (pd.read_csv(train_csv))
    table = pd.concat(tables)
    semantics = table["semantics"]
    file_ids = [x.split("/")[-1] for x in list(table["wav"])]
    actions = []
    scenarios = []
    intents = []
    for sem in semantics : 
        scenario, action, intent = get_scenario_action(sem)
        actions.append(action)
        scenarios.append(scenario)
        intents.append(intent)
    return  file_ids, scenarios, actions, intents

train_all = create_dict([train_csv, train_synthetic])
test_all = create_dict([test_csv])
#test_all = create_dict([train_csv, train_synthetic])
dev_all = create_dict([dev_csv])
train_dataset = tf.data.Dataset.from_tensor_slices(train_all)
test_dataset = tf.data.Dataset.from_tensor_slices(test_all)
dev_dataset = tf.data.Dataset.from_tensor_slices(dev_all)


TEST_BATCH_SPLIT = 3
# A single test batch.
DEV_BATCH = next(
    dev_dataset.take(TEST_BATCH_SPLIT)
    .apply(functools.partial(preprocess, batch_size=TEST_BATCH_SPLIT))
    .as_numpy_iterator()
)

TEST_BATCHES = (
    test_dataset.skip(TEST_BATCH_SPLIT)
    .apply(functools.partial(preprocess))
    .prefetch(tf.data.AUTOTUNE)
    .as_numpy_iterator()
)

# An iterator of training batches.
TRAIN_BATCHES = (
    train_dataset.skip(TEST_BATCH_SPLIT)
    .apply(functools.partial(preprocess))
    .prefetch(tf.data.AUTOTUNE)
    .as_numpy_iterator()
)

checkpoint_dir = "first_checkpoint"
if not os.path.exists(checkpoint_dir) : 
    os.makedirs(checkpoint_dir)

options = orbax.checkpoint.CheckpointManagerOptions(save_interval_steps=2, max_to_keep=2)
mngr = orbax.checkpoint.CheckpointManager(
    checkpoint_dir, orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()), options)

with open('/export/corpora/representations/SLURP/test_out.pickle', 'rb') as f:
  lattice_config, lattice_params = pickle.load(f)

lattice = lattice_config.build()

def untokenize(hyp_labels):
  return ''.join([GRAPHEME_SYMBOLS[i] for i in hyp_labels])

import torch
from torchdata.datapipes.iter import FileLister, FileOpener
class LSTMEncoder(nn.Module):
  """A stack of unidirectional LSTMs."""

  hidden_size: int
  num_layers: int

  @nn.compact
  def __call__(self, xs):
    """Encodes the inputs.

    Args:
      xs: [batch_size, max_num_frames, feature_size] input sequences.

    Returns:
      [batch_size, max_num_frames, hidden_size] output sequences.
    """
    # Flax allows randomness in initializing the hidden state. However LSTM's
    # initial hidden state is deterministic, so we just need a dummy RNG here.
    dummy_rng = jax.random.PRNGKey(0)
    init_carry = nn.OptimizedLSTMCell.initialize_carry(
        rng=dummy_rng, batch_dims=xs.shape[:1], size=self.hidden_size
    )
    # A stack of num_layers LSTMs.
    for _ in range(self.num_layers):
      # nn.scan() when passed a class returns another class like object. Thus
      # nn.scan(nn.OptimizedLSTMCell, ...)() constructs such a new instance of
      # such a "class". Then, we invoke the __call__() method of this object
      # by passing (init_carry, xs).
      #
      # The __call__() method of the nn.scan()-transformed "class" loops through
      # the input whereas the original class takes one step.
      _, xs = nn.scan(
          nn.OptimizedLSTMCell,
          variable_broadcast='params',
          split_rngs={'params': False},
          in_axes=1,
          out_axes=1,
      )()(init_carry, xs)
    return xs

class SimpleBiLSTM(nn.Module):
  """A simple bi-directional LSTM."""
  hidden_size: int
  def setup(self):
    #self.forward_lstm = nn.RNN(nn.LSTMCell(self.hidden_size), return_carry=True)
    self.forward_lstm = LSTMEncoder(self.hidden_size, 1)
    self.backward_lstm = LSTMEncoder(self.hidden_size, 1)

  def __call__(self, embedded_inputs, lengths):
    # Forward LSTM.
    forward_outputs = self.forward_lstm( embedded_inputs)

    # Backward LSTM.
    reversed_inputs = flip_sequences(embedded_inputs, lengths)
    backward_outputs = self.backward_lstm(reversed_inputs)
    backward_outputs = flip_sequences(backward_outputs, lengths)

    # Concatenate the forwardand backward representations.
    outputs = jnp.concatenate([forward_outputs, backward_outputs], -1)
    return outputs

class KeysOnlyMlpAttention(nn.Module):
  """Computes MLP-based attention scores based on keys alone, without a query.

  Attention scores are computed by feeding the keys through an MLP. This
  results in a single scalar per key, and for each sequence the attention
  scores are normalized using a softmax so that they sum to 1. Invalid key
  positions are ignored as indicated by the mask. This is also called
  "Bahdanau attention" and was originally proposed in:
  ```
  Bahdanau et al., 2015. Neural Machine Translation by Jointly Learning to
  Align and Translate. ICLR. https://arxiv.org/abs/1409.0473
  ```

  Attributes:
    hidden_size: The hidden size of the MLP that computes the attention score.
  """
  hidden_size: int

  @nn.compact
  def __call__(self, keys,mask):
    """Applies model to the input keys and mask.

    Args:
      keys: The inputs for which to compute an attention score. Shape:
        <float32>[batch_size, seq_length, embeddings_size].
      mask: A mask that determines which values in `keys` are valid. Only
        values for which the mask is True will get non-zero attention scores.
        <bool>[batch_size, seq_length].

    Returns:
      The normalized attention scores. <float32>[batch_size, seq_length].
    """
    hidden = nn.Dense(self.hidden_size, name='keys', use_bias=False)(keys)
    energy = nn.tanh(hidden)
    scores = nn.Dense(1, name='energy', use_bias=False)(energy)
    scores = scores.squeeze(-1)  # New shape: <float32>[batch_size, seq_len].
    scores = jnp.where(mask, scores, -jnp.inf)  # Using exp(-inf) = 0 below.
    scores = nn.softmax(scores, axis=-1)

    # Captures the scores if 'intermediates' is mutable, otherwise does nothing.
    self.sow('intermediates', 'attention', scores)

    return scores


class Model(nn.Module):
  # Size for encoder LSTMs, the output context LSTM, the joint weight function.
  hidden_size: int = 256
  num_encoder_layers: int = 1
  num_scenarios = scen_count
  num_actions = act_count
  dropout_rate = 0.1
  def setup(self):
    # Putting all pieces together to build the RecoginitionLattice.
    self.lattice =lattice 
    self.Downstream_Encoder = SimpleBiLSTM(self.hidden_size)
    self.scenarios_head = nn.Dense(self.num_scenarios)
    self.actions_head = nn.Dense(self.num_actions)
    self.keys_only_mlp_attention = KeysOnlyMlpAttention(512)
    self.dropout_layer = nn.Dropout(rate=self.dropout_rate)
    self.regrouping_head = nn.Dense(1)
  def __call__(self, batch):
    features = batch["encoder_frames"]
    # The __call__() method of RecognitionLattice returns [batch_size] per-
    # This needs to be changed 
    cache = self.lattice.build_cache()
    blank, vocab = self.lattice.weight_fn(cache, features)
    blank_expanded = jnp.expand_dims(blank, axis =3)
    full_lattice = jnp.concatenate([blank_expanded, vocab], axis = 3)
    regrouped_lattice = self.regrouping_head(full_lattice)
    regrouped_lattice = jnp.squeeze(regrouped_lattice, axis=3) # TODO Add the axis here
    regrouped_lattice = jnp.exp(regrouped_lattice)
    encoded_lattice=  self.Downstream_Encoder(regrouped_lattice, batch["num_frames"])
    mask = sequence_mask(batch["num_frames"], encoded_lattice.shape[1])
    attention = self.keys_only_mlp_attention(encoded_lattice, mask)

    # Summarize the inputs by taking their weighted sum using attention scores.
    context = jnp.expand_dims(attention, 1) @ encoded_lattice
    context = context.squeeze(1)  # <float32>[batch_size, encoded_inputs_size]
    #context = self.dropout_layer(context, deterministic=deterministic)

    scenarios = self.scenarios_head(context)
    actions = self.actions_head(context)
    return scenarios, actions

def compute_accuracies(actions, scenarios, batch):
    actions_results = actions ==batch["action"]
    scenarios_results = scenarios == batch["scenario"]
    intents = actions_results * scenarios_results
    return {"actions" : jnp.mean(actions_results),
            "scenarios" : jnp.mean(scenarios_results),
            "intents" : jnp.mean(intents)}

def train_and_eval(
    TEST_BATCHES,
    train_batches,
    model,
    step,
    optimizer=optax.adam(2e-4),
    num_steps=100000,
    num_steps_per_eval=2000,
    num_eval_steps=500
):
  # Initialize the model parameters using a fixed RNG seed. Flax linen Modules
  # need to know the shape and dtype of its input to initialize the parameters,
  # we thus pass it the test batch.
  if step is None : 
      params = model.init(jax.random.PRNGKey(0), DEV_BATCH)
      opt_state = optimizer.init(params)
  else : 
      print(f" step loaded : {step}")
      params = model.init(jax.random.PRNGKey(0), DEV_BATCH)
      opt_state = optimizer.init(params)
      params,opt_state = mngr.restore(step, items = [params, opt_state])

  # jax.jit compiles a JAX function to speed up execution.
  # `donate_argnums=(0, 1)` means we know that the input `params` and
  # `opt_state` won't be needed after calling `train_step`, so we donate them
  # and allow JAX to use their memory for storing the output.
  @functools.partial(jax.jit, donate_argnums=(0, 1))
  def train_step(params, opt_state, batch):
    # Compute the loss value and the gradients.
    def loss_fn(params) : 
        scenarios_logits, actions_logits = model.apply(params,batch) 
        loss_scen = optax.softmax_cross_entropy_with_integer_labels(scenarios_logits, batch["scenario"])
        loss_actions = optax.softmax_cross_entropy_with_integer_labels(actions_logits, batch["action"])
        return jnp.mean(loss_actions) + jnp.mean(loss_scen)
    loss, grads = jax.value_and_grad(loss_fn)(params)

    # Compute the actual updates based on the optimizer state and the gradients.
    updates, opt_state = optimizer.update(grads, opt_state, params)
    # Apply the updates.
    params = optax.apply_updates(params, updates)
    return params, opt_state, {'loss': loss}

  # We are not passing additional arguments to jax.jit, so it can be used
  # directly as a function decorator.
  @jax.jit
  def eval_step(params, batch):
    scenarios_logits, actions_logits = model.apply(params, batch)
    # Test accuracy.
    scenarios = jnp.argmax(scenarios_logits, axis = 1)
    actions = jnp.argmax(actions_logits, axis = 1)
    #Compute accuracies 
    return compute_accuracies(actions, scenarios, batch)  

  num_done_steps = 0
  while num_done_steps < num_steps:
    for step in tqdm(range(num_steps_per_eval)):
      next_batch = next(train_batches)
      params, opt_state, train_metrics = train_step(
          params, opt_state, next(train_batches)
      )
    mngr.save(num_done_steps,[params, opt_state])

    eval_metrics = {"actions": [],"scenarios" : [], "intents" :[] }
    for _ in tqdm(range(num_eval_steps)) : 

        test_batch = next(TEST_BATCHES)
        eval_metrics_step = eval_step(params, test_batch)
        for i in eval_metrics : 
            eval_metrics[i].append(eval_metrics_step[i])
       

    num_done_steps += num_steps_per_eval
    print(f'step {num_done_steps}\ttrain {train_metrics}')
    for i in eval_metrics : 
        print(f" {i} : {jnp.mean(jnp.array(eval_metrics[i]))}")
 

model = Model()
step = mngr.latest_step()
#import pdb; pdb.set_trace()
train_and_eval(TEST_BATCHES, TRAIN_BATCHES, model, step)
