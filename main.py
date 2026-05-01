import os
import torch
import rlcard
import numpy as np
from rlcard.agents.dmc_agent import DMCTrainer
from rlcard.envs.registration import register
from rlcard.agents.dmc_agent.model import DMCModel
from rlcard.agents.dmc_agent.model import DMCAgent
from rlcard.utils import tournament
import math
import random

register(env_id='tien-len', entry_point='envs.tienlen_env:TienLenEnv')
env = rlcard.make('tien-len')


orig_init = DMCAgent.__init__
def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.use_raw = True
DMCAgent.__init__ = new_init


# Agents
class TrainedAgent:
    def __init__(self, model):
        self.model = model
    def step(self, state):
        # Option 2: Score all legal action features and pick the max
        obs = torch.FloatTensor(state['obs']).unsqueeze(0)
        legal_actions = state['legal_actions']
        best_action, max_q = None, -float('inf')

        for action, feature in legal_actions.items():
            feat = torch.FloatTensor(feature).unsqueeze(0)
            with torch.no_grad():
                q_value = self.model(obs, feat)
            if q_value > max_q:
                max_q = q_value
                best_action = action
        return best_action

    def eval_step(self, state):
        return self.step(state), {}


class MoveSequence(list):
    def __init__(self, score, sequence_or_node):
        if isinstance(sequence_or_node, list):
            super().__init__(sequence_or_node)
        else:
            super().__init__([sequence_or_node])
        self.score = score

class HeuristicSearchAgent:
    def __init__(self, judger, pruning_threshold=0.2):
        self.judger = judger
        self.PRUNING_THRESHOLD = pruning_threshold

    def evaluate_move(self, hand, move_played, state, moves):
        """ Direct port of Java evaluateMove logic """
        score = 10 * math.log(1 + len(moves)) if len(moves) > 0 else 0
        if not hand: score += 1000
        score += math.pow(1.5, (13 - len(hand)))

        if hand:
            rank_sum = sum(c[0] * 4 + c[1] for c in hand)
            score += (rank_sum / len(hand)) * 5

        if moves:
            score += sum(len(m) for m in moves) / len(moves)

        move_type, _ = self.judger.get_type(move_played)
        if move_played:
            # Re-calculating mobility after playing this move
            # state here is the stackState (none, pig, etc.)
            future_moves = self.judger._get_all_types(hand)
            for m in future_moves:
                score += math.log(len(m)) if state == "NONE" else len(m)

        if move_type == "HANG" and state in ["PIG", "SAME"]:
            score += 250
        return score

    def minimax(self, hand, move, state, depth, acc_score, best_score_param):
        # BUG FIX 1: Use list removal to avoid tuple identity issues
        hand_copy = list(hand)
        for card in move:
            if card in hand_copy:
                hand_copy.remove(card)

        new_move_state, _ = self.judger.get_type(move)
        # Assuming get_all_types is the 'getMoves' call from Java
        next_moves = self.judger._get_all_types(hand_copy)

        current_score = self.evaluate_move(hand_copy, move, state, next_moves)

        # BUG FIX 2: Pruning logic check
        # Mirrors Java: if (currentScore < bestScore * PRUNING_THRESHOLD)
        if depth == 0 or not next_moves or current_score < best_score_param * self.PRUNING_THRESHOLD:
            # Java: return new MoveSequence(accumulatedScore + (currentScore / Math.pow(0.9, depth)) * (depth + 1), move);
            final_score = acc_score + (current_score / (0.9**depth)) * (depth + 1)
            return MoveSequence(final_score, move)

        max_val = -float('inf')
        best_seq = []

        for nxt in next_moves:
            # Passing max_val as the new best_score_param for the next depth
            eval_res = self.minimax(hand_copy, nxt, new_move_state, depth - 1,
                                    acc_score + current_score / (0.9**depth), max_val)

            if eval_res.score > max_val:
                max_val = eval_res.score
                best_seq = list(eval_res)

        best_seq.insert(0, move)
        return MoveSequence(max_val, best_seq)

    def step(self, state):
        hand = state['raw_obs']['hand']
        stack_state = state['raw_obs']['state']
        legal_moves = [m for m in state['raw_legal_actions'] if m] # Skip 'Pass'

        if not legal_moves:
            return ()

        best_move = None
        best_val_obj = None # BUG FIX 3: Clearer initialization

        for m in legal_moves:
            # depth=2 as in your Java getMove(state, peek, 2)
            res = self.minimax(hand, m, stack_state, 2, 0, -float('inf'))
            res.score /= 3.0 # depth + 1

            if best_val_obj is None or res.score > best_val_obj.score:
                best_val_obj = res
                best_move = m

        # Compare against baseline (not playing anything)
        current_moves = self.judger._get_all_types(hand)
        baseline = self.evaluate_move(hand, (), "NONE", current_moves)

        if best_val_obj and best_val_obj.score > baseline:
            return best_move
        return ()

    def eval_step(self, state):
        return self.step(state), {}

class NaiveTienLenAgent:
    """
    Naive implementation: Always plays most cards possible, smallest first.
    If an opponent is about to win, it switches to playing high cards.
    """
    def __init__(self):
        pass

    def step(self, state):
        legal_actions = state['raw_legal_actions'] # List of card tuples
        if not legal_actions or legal_actions == [()]:
            return ()

        # Remove 'Pass' from consideration for sorting logic
        playable = [a for a in legal_actions if a != ()]
        if not playable: return ()

        # Java logic: Sort by size (desc), then rank (asc), then suit (desc)
        # Note: action is a tuple of (rank, suit)
        playable.sort(key=lambda move: (
            -len(move),               # sizeCompare (descending)
            move[0][0],               # rankCompare (ascending)
            -move[-1][1]              # suitCompare (descending)
        ))

        # Check if someone is about to win (Java: someoneAboutToWin)
        # state['others_hand_count'] from our Game.get_state
        someone_about_to_win = any(count*13 == 1 for count in state['obs'][156:159])

        # Java logic: If same state and someone winning, play the highest legal same-rank combo
        if someone_about_to_win:
            return playable[-1]

        return playable[0]

    def eval_step(self, state):
        return self.step(state), {}


class TienLenRandomAgent:
    """
    A random agent that works with Option 2 (Dynamic Actions).
    Instead of picking an ID, it picks a random card tuple.
    """
    def __init__(self):
        self.use_raw = True
        pass

    def step(self, state):
        """
        state['legal_actions'] is a dict where keys are card tuples
        e.g., { ((3,0), (3,1)): [feature_vector], ... }
        """
        # 1. Get all currently legal card combinations
        legal_moves = list(state['legal_actions'].keys())

        # 2. Pick one at random
        if not legal_moves:
            return () # Fallback to Pass

        return random.choice(legal_moves)

    def eval_step(self, state):
        return self.step(state), {}

def train():
    env = rlcard.make('tien-len')
    trainer = DMCTrainer(
        env,
        cuda='0',
        xpid='rl-tien-len',
        savedir='models',
        num_actors=5,
    )
    trainer.start()

def load_model():
    env = rlcard.make('tien-len')

    model = DMCModel(env.state_shape, [52])
    checkpoint = torch.load('models/rl-tien-len/model.tar', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set to evaluation mode

    return TrainedAgent(model)


def evaluate(trained_agent):
    env = rlcard.make('tien-len')
    env.set_agents([trained_agent, NaiveTienLenAgent(), HeuristicSearchAgent(env.game.judger), TienLenRandomAgent()])
    results = tournament(env, 1000)
    print(f"Tournament Results (Avg Payoffs): {results}")


def play_test_game(trained_agent):
    env = rlcard.make('tien-len')
    env.set_agents([trained_agent, NaiveTienLenAgent(), HeuristicSearchAgent(env.game.judger), TienLenRandomAgent()])

    state, player_id = env.reset()
    print("--- Game Start ---")

    while not env.is_over():
        # Get action from the current player's agent
        current_agent = env.agents[player_id]
        action, _ = current_agent.eval_step(state)

        # Display the play
        move_desc = action if action else "PASS"
        print(f"Player {player_id} plays: {move_desc}")

        # Advance environment
        state, player_id = env.step(action)

    print("--- Game Over ---")
    print(f"Final Payoffs: {env.get_payoffs()}")


if not os.path.exists('models'): os.makedirs('models')
train()
agent = load_model()

evaluate(agent)

play_test_game(agent)