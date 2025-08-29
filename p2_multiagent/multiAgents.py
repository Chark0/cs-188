# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Base score from game state
        score = successorGameState.getScore()

        heavy_mod = 10
        medium_mod = 5 
        light_mod = 2 

        "*** YOUR CODE HERE ***"
        # food considerations
        foodList = newFood.asList()
        if foodList:
            closestFoodDist = min(manhattanDistance(newPos, food) for food in foodList)
            score += heavy_mod / (closestFoodDist + 1)  # Reward being closer to food
            score -= len(foodList) * medium_mod  # Penalty for remaining food

        # ghost considerations
        ghost_weight = 50
        for i, ghost in enumerate(newGhostStates):
            ghostPos = ghost.getPosition()
            ghostDist = manhattanDistance(newPos, ghostPos)
            g_dist_reciprocal = (1 / max(ghostDist, 0.1))

            if newScaredTimes[i] > 0:
                # Chase scared ghosts for points
                if ghostDist > 0:
                    score += medium_mod * ghost_weight * g_dist_reciprocal # slight bonus reward
            else:
                # Avoid active ghosts
                if ghostDist <= 1:
                    score -= heavy_mod * ghost_weight * g_dist_reciprocal  # Heavy penalty
                elif ghostDist <= 3:
                    score -= ghost_weight * g_dist_reciprocal  # normalized penalty
        
        # Small penalty for stopping (encourages movement)
        if action == 'Stop':
            score -= medium_mod

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # get ghost indices
        ghostIdx = [i for i in range(1, gameState.getNumAgents())]

        def minValue(state, depth, ghost):
            """
            Returns the minimum value of the game state for the ghost agent.
            """
            if terminalState(state, depth):
                return self.evaluationFunction(state)

            minValue = float('inf')
            legalActions = state.getLegalActions(ghost)

            for action in legalActions:
                if ghost == ghostIdx[-1]:
                    value = min(value, maxValue(state.generateSuccessor(ghost, action), depth + 1))
                else:
                    value = min(value, minValue(state.generateSuccessor(ghost, action), depth, ghost + 1))

            return value
        
        def maxValue(state, depth):
            """
            Returns the maximum value of the game state for Pacman.
            """
            if terminalState(state, depth):
                return self.evaluationFunction(state)

            maxValue = float('-inf')
            legalActions = state.getLegalActions(0)

            for action in legalActions:
                successorState = state.generateSuccessor(0, action)
                value = minValue(successorState, depth, ghostIdx[0])
                
                if value > maxValue:
                    maxValue = value

            return maxValue
        
        def terminalState(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth
        
        
        res = []
        for action in gameState.getLegalActions(0):
            successor_state = gameState.generateSuccessor(0, action)
            min_score = minValue(successor_state, 0, ghostIdx[0])
            tuple_result = (action, min_score)
            res.append(tuple_result)
        
        # Find the action with the maximum score sort the result and take the first action
        res.sort(key=lambda x: x[1], reverse=True)
        return res[0][0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        """
        Cleaner unified alpha-beta implementation
        """
        
        def minimax_alpha_beta(state, depth, agent_index, alpha, beta):
            # Terminal test
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            # Calculate next agent and depth
            num_agents = state.getNumAgents()
            next_agent = (agent_index + 1) % num_agents
            next_depth = depth + 1 if next_agent == 0 else depth
            
            legal_actions = state.getLegalActions(agent_index)
            
            if agent_index == 0:  # Pacman (maximizing)
                value = float('-inf')
                for action in legal_actions:
                    successor = state.generateSuccessor(agent_index, action)
                    value = max(value, minimax_alpha_beta(successor, next_depth, next_agent, alpha, beta))
                    
                    # Alpha-beta pruning
                    if value > beta:
                        return value  # Beta cutoff
                    alpha = max(alpha, value)
                    
                return value
                
            else:  # Ghost (minimizing)
                value = float('inf')
                for action in legal_actions:
                    successor = state.generateSuccessor(agent_index, action)
                    value = min(value, minimax_alpha_beta(successor, next_depth, next_agent, alpha, beta))
                    
                    # Alpha-beta pruning
                    if value < alpha:
                        return value  # Alpha cutoff
                    beta = min(beta, value)
                    
                return value
        
        # Find best action
        best_action = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = minimax_alpha_beta(successor, 0, 1, alpha, beta)
            
            if value > best_value:
                best_value = value
                best_action = action
                
            alpha = max(alpha, value)  # Update for pruning
        
        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(state, depth, agentIndex):
            """
            Expectimax algorithm: Max for Pacman, Expected value for random ghosts
            """
            # Terminal test
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            # Calculate next agent and depth
            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth
            
            legalActions = state.getLegalActions(agentIndex)
            
            if agentIndex == 0:  # Pacman (maximizing player)
                # Note no change here from minimax, we still look to maximize
                value = float('-inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, expectimax(successor, nextDepth, nextAgent))
                return value
                
            else:  # Ghost (random player)
                # CHANGED: Instead of minimizing, calculate EV
                totalValue = 0.0
                numActions = len(legalActions)
                
                # Each action has equal probability: 1/numActions
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    actionValue = expectimax(successor, nextDepth, nextAgent)
                    totalValue += actionValue
                
                # Return average value (expected value)
                return totalValue / numActions
        
        # Find best action for Pacman
        bestAction = None
        bestValue = float('-inf')
        
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = expectimax(successor, 0, 1)
            
            if value > bestValue:
                bestValue = value
                bestAction = action
        
        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Add to score when close to food, and capsules - when chosts are close
    then ghosts when they are scared
    """

    # Terminal states - handle win/lose immediately
    if currentGameState.isWin():
        return float('inf')  # Winning is infinitely good!
    if currentGameState.isLose():
        return float('-inf')  # Losing is infinitely bad!
    
    
    # get state info
    score = currentGameState.getScore()
    pacmanPos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    foodList = currentGameState.getFood().asList()
    capsuleList = currentGameState.getCapsules()

    heavy_mod = 100
    medium_mod = 50
    light_mod = 20

    ghost_weight = 100

    # food evaluation
    if foodList:
        closestFoodDist = min(manhattanDistance(pacmanPos, food) for food in foodList)
        score += heavy_mod / (closestFoodDist + 1)  # Reward being closer to food
        score -= len(foodList) * heavy_mod  # Penalty for remaining food

    # ghost evaluation
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        ghostDist = manhattanDistance(pacmanPos, ghostPos)
        g_dist_reciprocal = (1 / max(ghostDist, 0.1))

        if ghost.scaredTimer > 0:
            # Chase scared ghosts for points
            if ghostDist > 0:
                score += heavy_mod * ghost_weight * g_dist_reciprocal # slight bonus reward
        else:
            # Avoid active ghosts
            if ghostDist <= 1:
                score -= heavy_mod * ghost_weight * g_dist_reciprocal  # Heavy penalty
            elif ghostDist <= 3:
                score -= ghost_weight * g_dist_reciprocal  # normalized penalty
    
    # capsule evaluation
    for capsule in capsuleList:
        capsuleDistance = manhattanDistance(pacmanPos, capsule)
        
        # Encourage getting capsules when ghosts are nearby
        nearbyGhosts = sum(1 for ghost in ghostStates 
                          if manhattanDistance(pacmanPos, ghost.getPosition()) <= 5 
                          and ghost.scaredTimer == 0)
        
        if nearbyGhosts > 0:
            # More valuable when ghosts are threats
            score += (ghost_weight * nearbyGhosts) / (capsuleDistance + 1)
        else:
            # Still valuable but less urgent
            score += ghost_weight / (capsuleDistance + 1)

    return score
    

# Abbreviation
better = betterEvaluationFunction
