#Note: All hyperparameters and environment conditions have been changed in this example from their original. The below code is meant to showcase the methodology in solving machine learning
#and reinforcement learning problems.

#The following algorithm shows different setups of Deep Q-Learning. In this scenario, a hypothetical car collision between two cars is studied. Four different implementation of Deep Q-Learning
#have been implemented. Each of such implementation will assume different intentions of each car, and thus, successful convergence of the machine learning model will depend on the use of the
#right learning method.

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from cvxopt import matrix, solvers
import time

# The below class is a simplification of an environment produced from comprehensive data. The model is closely mirrored, but not exactly, by this class.
class Collision:
    def __init__(self):
        self.rightOfWay = 1
        self.bypass = [0, 3] # B bypass position
        self.pos = [np.array([0, 2]), np.array([0, 1])] #car pos


    def move(self, actions):

        # Init Scores
        scores = np.array([0, 0])
        
        # Five Legal Actions
        allowedActions = [[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]]
        if actions[0] not in range(0,5) or actions[1] not in range(0,5):
            print('Invalidmove')
            return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1], self.rightOfWay], scores, done

        # Randomly decide first mover. 0 is A, 1 is B.
        firstMover = np.random.choice([0, 1], 1)[0]
        secondMover = 1 - firstMover


        nextPos = self.pos.copy()
        done = 0

        # First car moves
        nextPos[firstMover] = self.pos[firstMover] + allowedActions[actions[firstMover]]

        # If collision then
        if (nextPos[firstMover] == self.pos[secondMover]).all():
            # Exchange rightOfWay
            if self.rightOfWay == firstMover:
                self.rightOfWay = secondMover

        # no collision
        elif nextPos[firstMover][0] in range(0,2) and nextPos[firstMover][1] in range(0,4):
            self.pos[firstMover] = nextPos[firstMover]

            # Us scored
            if self.pos[firstMover][1] == self.bypass[firstMover] and self.rightOfWay == firstMover:
                scores = ([1, -1][firstMover]) * np.array([100, -100])
                done = 1
                return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1], self.rightOfWay], scores, done

            # Them scored
            elif self.pos[firstMover][1] == self.bypass[secondMover] and self.rightOfWay == firstMover:
                scores = ([1, -1][firstMover]) * np.array([-100, 100])
                done = 1
                return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1], self.rightOfWay], scores, done


        # Second car moves
        nextPos[secondMover] = self.pos[secondMover] + allowedActions[actions[secondMover]]

        # If collision then
        if (nextPos[secondMover] == self.pos[firstMover]).all():
            # Exchange rightOfWay
            if self.rightOfWay == secondMover:
                self.rightOfWay = firstMover

        # No collision
        elif nextPos[secondMover][0] in range(0,2) and nextPos[secondMover][1] in range(0,4):
            self.pos[secondMover] = nextPos[secondMover]

            # Us scored
            if self.pos[secondMover][1] == self.bypass[secondMover] and self.rightOfWay == secondMover:
                scores = ([1, -1][secondMover]) * np.array([100, -100])
                done = 1
                return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1], self.rightOfWay], scores, done

            # Them scored
            elif self.pos[secondMover][1] == self.bypass[firstMover] and self.rightOfWay == secondMover:
                scores = np.array([-100, 100]) * [1, -1][secondMover]
                done = 1
                return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1], self.rightOfWay], scores, done


        return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1], self.rightOfWay], scores, done

solvers.options['show_progress'] = False

def correlatedQ():

    np.random.seed(123456)
  
    alpha = 0.995
    alpha_min = 0.001
    alpha_decay = 10 ** (np.log10(0.001)/1000000)
    gamma = 0.9
    epsilon_min = 0.001
    epsilon_decay = 10 ** (np.log10(0.001)/1000000)
    
    # Q_tables of car A and car B: 8 (pos for A) * 8 (pos for B) * 2 (rightOfWay) * 5 (valid actions A) * 5 (valid actions B)
    Q1 = np.ones((8, 8, 2, 5, 5)) * 1.0
    Q2 = np.ones((8, 8, 2, 5, 5)) * 1.0

    # value of states, only depends on pos of cars and possession of rightOfWay
    V1 = np.ones((8, 8, 2)) * 1.0
    V2 = np.ones((8, 8, 2)) * 1.0

    # shared joint policy
    Pi = np.ones((8, 8, 2, 5, 5)) * 1/25

    errorOut = []

   
    def qAction(Pi, state, i):
        epsilon = epsilon_decay ** i

        if np.random.random() < epsilon:
            index = np.random.choice(np.arange(25), 1)
            return np.array([index // 5, index % 5]).reshape(2)

        else:
            index = np.random.choice(np.arange(25), 1, p=Pi[state[0]][state[1]][state[2]].reshape(25))
            return np.array([index // 5, index % 5]).reshape(2)


    def solveCeq(Q1, Q2, state):
        # car A state subset
        Qstates = Q1[state[0]][state[1]][state[2]]
        s = block_diag(Qstates - Qstates[0, :], Qstates - Qstates[1, :], Qstates - Qstates[2, :], Qstates - Qstates[3, :], Qstates - Qstates[4, :])
        # Row Index has the same result as performing as_strided function.
        rowidx = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23]
        param1 = s[rowidx, :]

        # car B state subset
        Qstates = Q2[state[0]][state[1]][state[2]]
        s = block_diag(Qstates - Qstates[0, :], Qstates - Qstates[1, :], Qstates - Qstates[2, :], Qstates - Qstates[3, :], Qstates - Qstates[4, :])
        # Column Index is a square matrix function
        colidx = [0, 5, 10, 15, 20, 1, 6, 11, 16, 21, 2, 7, 12, 17, 22, 3, 8, 13, 18, 23, 4, 9, 14, 19, 24]
        param2 = s[rowidx, :][:, colidx]

        #Linear Programming is defined here.
        c = matrix((Q1[state[0]][state[1]][state[2]] + Q2[state[0]][state[1]][state[2]].T).reshape(25))
        G = matrix(np.append(np.append(param1, param2, axis=0), -np.eye(25), axis=0))
        h = matrix(np.zeros(65) * 0.0)
        A = matrix(np.ones((1, 25)))
        b = matrix(1.0)

        try:
            sol = solvers.lp(c=c, G=G, h=h, A=A, b=b)
            if sol['x'] is not None:
                prob = np.abs(np.array(sol['x']).reshape((5, 5))) / sum(np.abs(sol['x']))
                val1 = np.sum(prob * Q1[state[0]][state[1]][state[2]])
                val2 = np.sum(prob * Q2[state[0]][state[1]][state[2]].T)
            else:
                prob = None
                val1 = None
                val2 = None
        except:
            #print("err")
            prob = None
            val1 = None
            val2 = None

        return prob, val1, val2


    start_time = time.time()
    i = 0
    while i < 1000000:
        collision = Collision()
        state = [collision.pos[0][0] * 4 + collision.pos[0][1], collision.pos[1][0] * 4 + collision.pos[1][1], collision.rightOfWay]
        done = 0
        j = 0
        while not done and j <= 100:
            if i % 10000 == 0:
                print('\rstep {}\t Time: {:.2f} \t Alpha: {:.3f}'.format(i, time.time() - start_time, alpha))

            i, j = i+1, j+1

            before = Q1[2][1][1][2][4]

            #Generate Action
            actions = qAction(Pi, state, i)

            nextState, rewards, done = collision.move(actions)
            alpha = alpha_decay ** i

            # Q1 Update
            Q1[state[0]][state[1]][state[2]][actions[0]][actions[1]] = (1 - alpha) * Q1[state[0]][state[1]][state[2]][actions[0]][actions[1]] + alpha * (rewards[0] + gamma * V1[nextState[0]][nextState[1]][nextState[2]])

            # Q2 Update
            Q2[state[0]][state[1]][state[2]][actions[1]][actions[0]] = (1 - alpha) * Q2[state[0]][state[1]][state[2]][actions[1]][actions[0]] + alpha * (rewards[1] + gamma * V2[nextState[0]][nextState[1]][nextState[2]].T)
            prob, val1, val2 = solveCeq(Q1, Q2, state)

            # Null Update
            if prob is not None:
                Pi[state[0]][state[1]][state[2]] = prob
                V1[state[0]][state[1]][state[2]] = val1
                V2[state[0]][state[1]][state[2]] = val2
            state = nextState

           
            after = Q1[2][1][1][2][4]

           
            errorOut.append(np.abs(after - before))

    return errorOut

ceqerr = correlatedQ()

plt.plot(errors, linestyle='-', linewidth=0.6)
plt.title("CEQ")
plt.ylim(0, 0.5)
plt.xlabel('Simulation')
plt.ylabel('Difference')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
plt.show()


def FoeQ():

    np.random.seed(123456)
  
    alpha = 0.995
    alpha_min = 0.001
    alpha_decay = 10 ** (np.log10(0.001)/1000000)
    gamma = 0.9
    epsilon_min = 0.001
    epsilon_decay = 10 ** (np.log10(0.001)/1000000)

    # Q_tables of car A and car B: 8 (pos for A) * 8 (pos for B) * 2 (rightOfWay) * 5 (valid actions A) * 5 (valid actions B)
    Q1 = np.ones((8, 8, 2, 5, 5)) * 1.0
    Q2 = np.ones((8, 8, 2, 5, 5)) * 1.0

    # value of states, only depends on pos of cars and possession of rightOfWay
    V1 = np.ones((8, 8, 2)) * 1.0
    V2 = np.ones((8, 8, 2)) * 1.0

    # policy
    Pi1 = np.ones((8, 8, 2, 5)) * 1/5
    Pi2 = np.ones((8, 8, 2, 5)) * 1/5

    errorOut = []

    def generateAction(pi, state, i):
        epsilon = epsilon_decay ** i
        if np.random.random() < epsilon:
            return np.random.choice([0,1,2,3,4], 1)[0]
        else:
            return np.random.choice([0,1,2,3,4], 1, p=pi[state[0]][state[1]][state[2]])[0]

    def maxMin(Q, state):
        c = matrix([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        G = matrix(np.append(np.append(np.ones((5,1)), -Q[state[0]][state[1]][state[2]], axis=1), np.append(np.zeros((5,1)), -np.eye(5), axis=1), axis=0))
        h = matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        A = matrix([[0.0],[1.0], [1.0], [1.0], [1.0], [1.0]])
        b = matrix(1.0)
        sol = solvers.lp(c=c, G=G, h=h, A=A, b=b)
        return np.abs(sol['x'][1:]).reshape((5,)) / sum(np.abs(sol['x'][1:])), np.array(sol['x'][0])


    start_time = time.time()
    i = 0

    while i < 1000000:
        collision = Collision()
        state = [collision.pos[0][0] * 4 + collision.pos[0][1], collision.pos[1][0] * 4 + collision.pos[1][1], collision.rightOfWay]
        done = 0
        while not done:
            if i % 10000 == 0:
                print('\rstep {}\t Time: {:.2f} \t Alpha: {:.3f}'.format(i, time.time() - start_time, alpha))
            i += 1

            before = Q1[2][1][1][4][2]

            actions = [generateAction(Pi1, state, i), generateAction(Pi2, state, i)]

            nextState, rewards, done = collision.move(actions)

            # Q-learning update
            Q1[state[0]][state[1]][state[2]][actions[1]][actions[0]] = (1 - alpha) * Q1[state[0]][state[1]][state[2]][actions[1]][actions[0]] + alpha * (rewards[0] + gamma * V1[nextState[0]][nextState[1]][nextState[2]])

            # LP
            pi, val = maxMin(Q1, state)
            Pi1[state[0]][state[1]][state[2]] = pi
            V1[state[0]][state[1]][state[2]] = val

            # Q-learning update
            Q2[state[0]][state[1]][state[2]][actions[0]][actions[1]] = (1 - alpha) * Q2[state[0]][state[1]][state[2]][actions[0]][actions[1]] + alpha * (rewards[1] + gamma * V2[nextState[0]][nextState[1]][nextState[2]])

            # LP
            pi, val = maxMin(Q2, state)
            Pi2[state[0]][state[1]][state[2]] = pi
            V2[state[0]][state[1]][state[2]] = val
            state = nextState

            # compute err
            after = Q1[2][1][1][4][2]
            errorOut.append(np.abs(after - before))

            alpha = alpha_decay ** i

    return errorOut

foeqerr = FoeQ()

plt.plot(errors, linestyle='-', linewidth=0.6)
plt.title("FoeQ")
plt.ylim(0, 0.5)
plt.xlabel('Simulation')
plt.ylabel('Difference')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
plt.show()

def FriendQ():

    np.random.seed(123456)

    alpha = 0.995
    alpha_min = 0.001
    alpha_decay = 0.999995
    gamma = 0.9
    epsilon = 1.0
    epsilon_min = 0.001
    epsilon_decay = 0.999995

    # Q_tables of car A and car B: 8 (pos for A) * 8 (pos for B) * 2 (rightOfWay) * 5 (valid actions A) * 5 (valid actions B)
    Q1 = np.zeros((8, 8, 2, 5, 5))
    Q2 = np.zeros((8, 8, 2, 5, 5))

    errorOut = []

    def generateAction(Q, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice([0,1,2,3,4], 1)[0]

        maxidx = np.where(Q[state[0]][state[1]][state[2]] == np.max(Q[state[0]][state[1]][state[2]]))
        return maxidx[1][np.random.choice(range(len(maxidx[0])), 1)[0]]


    i = 0

    start_time = time.time()

    while i < 1000000:
        env = Collision()

        state = [env.pos[0][0] * 4 + env.pos[0][1], env.pos[1][0] * 4 + env.pos[1][1], env.rightOfWay]

        while True:
            if i % 10000 == 0:
                print('\rstep {}\t Time: {:.2f} \t Alpha: {:.3f}'.format(i, time.time() - start_time, alpha))

            before = Q1[2][1][1][4][2]

            
            actions = [generateAction(Q1,state,epsilon), generateAction(Q2,state,epsilon)]
           
            nextState, rewards, done = env.move(actions)

            alpha = 1 / (i / alpha_min / 1000000 + 1)

            i += 1

            # FriendQ
            if done:
                Q1[state[0]][state[1]][state[2]][actions[1]][actions[0]] = Q1[state[0]][state[1]][state[2]][actions[1]][actions[0]] + alpha * (rewards[0] - Q1[state[0]][state[1]][state[2]][actions[1]][actions[0]])

                Q2[state[0]][state[1]][state[2]][actions[0]][actions[1]] = Q2[state[0]][state[1]][state[2]][actions[0]][actions[1]] + alpha * (rewards[1] - Q2[state[0]][state[1]][state[2]][actions[0]][actions[1]])

                after = Q1[2][1][1][4][2]
                errorOut.append(abs(after - before))
                break

            else:
                Q1[state[0]][state[1]][state[2]][actions[1]][actions[0]] = Q1[state[0]][state[1]][state[2]][actions[1]][actions[0]] + alpha * (rewards[0] + gamma * np.max(Q1[nextState[0]][nextState[1]][nextState[2]]) - Q1[state[0]][state[1]][state[2]][actions[1]][actions[0]])

                Q2[state[0]][state[1]][state[2]][actions[0]][actions[1]] = Q2[state[0]][state[1]][state[2]][actions[0]][actions[1]] + alpha * (rewards[1] + gamma * np.max(Q2[nextState[0]][nextState[1]][nextState[2]]) - Q2[state[0]][state[1]][state[2]][actions[0]][actions[1]])
                state = nextState

                after = Q1[2][1][1][4][2]
                errorOut.append(abs(after - before))

            epsilon *= epsilon_decay
            epsilon = max(epsilon_min, epsilon)

    return errorOut

friendqerr = FriendQ()

plt.plot(errors, linestyle='-', linewidth=0.6)
plt.title("FriendQ")
plt.ylim(0, 0.5)
plt.xlabel('Simulation')
plt.ylabel('Difference')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
plt.show()

def Qlearning():

    np.random.seed(123456)
  
    alpha = 0.995
    alpha_min = 0.001
    alpha_decay = 0.999995
    gamma = 0.9
    epsilon = 1.0
    epsilon_min = 0.001
    epsilon_decay = 0.999995

    # Q_tables of car A and car B: 8 (pos for A) * 8 (pos for B) * 2 (rightOfWay) * 5 (valid actions)
    Q1 = np.zeros((8, 8, 2, 5))
    Q2 = np.zeros((8, 8, 2, 5))

    errorOut = []

    def generateAction(Q, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice([0,1,2,3,4], 1)[0]

        return np.random.choice(np.where(Q[state[0]][state[1]][state[2]] == max(Q[state[0]][state[1]][state[2]]))[0], 1)[0]

    i = 0

    start_time = time.time()

    while i < 1000000:
        env = Collision()
        state = [env.pos[0][0] * 4 + env.pos[0][1], env.pos[1][0] * 4 + env.pos[1][1], env.rightOfWay]

        while True:
            if i % 10000 == 0:
                print('\rstep {}\t Time: {:.2f} \t Alpha: {:.3f}'.format(i, time.time() - start_time, alpha))

            before = Q1[2][1][1][2]

            actions = [generateAction(Q1,state,epsilon), generateAction(Q2,state,epsilon)]
            nextState, rewards, done = env.move(actions)

            i += 1

            # Q-learning
            if done:
                Q1[state[0]][state[1]][state[2]][actions[0]] = Q1[state[0]][state[1]][state[2]][actions[0]] + alpha * (rewards[0] - Q1[state[0]][state[1]][state[2]][actions[0]])

                Q2[state[0]][state[1]][state[2]][actions[1]] = Q2[state[0]][state[1]][state[2]][actions[1]] + alpha * (rewards[1] - Q2[state[0]][state[1]][state[2]][actions[1]])
                after = Q1[2][1][1][2]
                errorOut.append(abs(after-before))
                break

            else:
                Q1[state[0]][state[1]][state[2]][actions[0]] = Q1[state[0]][state[1]][state[2]][actions[0]] + alpha * (rewards[0] + gamma * max(Q1[nextState[0]][nextState[1]][nextState[2]]) - Q1[state[0]][state[1]][state[2]][actions[0]])

                Q2[state[0]][state[1]][state[2]][actions[1]] = Q2[state[0]][state[1]][state[2]][actions[1]] + alpha * (rewards[1] + gamma * max(Q2[nextState[0]][nextState[1]][nextState[2]]) - Q2[state[0]][state[1]][state[2]][actions[1]])

                state = nextState

                after = Q1[2][1][1][2]
                errorOut.append(abs(after-before))

            epsilon *= epsilon_decay
            epsilon = max(epsilon_min, epsilon)

            alpha *= alpha_decay
            alpha = max(alpha_min, alpha)

    return errorOut

qerr = Qlearning()

plt.plot(errors, linestyle='-', linewidth=0.6)
plt.title("Q")
plt.ylim(0, 0.5)
plt.xlabel('Simulation')
plt.ylabel('Difference')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
plt.show()
