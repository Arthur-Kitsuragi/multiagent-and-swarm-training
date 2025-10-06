import numpy as np
import pyswarms as ps
import matplotlib.pyplot as plt

class NettingOptimizer:
    def __init__(self, obligations_matrix, threshold):
        """
        obligations_matrix: obligations_matrix[i,j] - debt of company i to company j,
        obligations_matrix[i][j] >= 0, obligations_matrix[i][i] = 0
        """
        self.obligations = np.array(obligations_matrix)
        self.n_companies = self.obligations.shape[0]
        self.calculate_net_positions()
        self.threshold = threshold

    def calculate_net_positions(self):

        self.receivables = self.obligations.sum(axis=0)

        self.payables = self.obligations.sum(axis=1)

        self.net_positions = self.receivables - self.payables

    def fitness_function(self, payment_schemes, payments_weight=50, count_weight=0.05, balance_weight=1000, double_weight=10):
        n_particles = payment_schemes.shape[0]
        fitness = np.zeros(n_particles)

        # max_amount = np.sum(np.abs(self.net_positions))
        # max_transactions = self.n_companies * (self.n_companies - 1)

        for p in range(n_particles):
            payment_matrix = payment_schemes[p].reshape(
                self.n_companies, self.n_companies)

            np.fill_diagonal(payment_matrix, 0)

            for i in range(self.n_companies):
                for j in range(i + 1, self.n_companies):
                    if payment_matrix[i, j] > 0 and payment_matrix[j, i] > 0:
                        net_payment = payment_matrix[i, j] - payment_matrix[j, i]

                        if net_payment > 0:
                            payment_matrix[i, j] = net_payment
                            payment_matrix[j, i] = 0
                        elif net_payment < 0:
                            payment_matrix[j, i] = -net_payment
                            payment_matrix[i, j] = 0
                        else:  # net_payment == 0
                            payment_matrix[i, j] = 0
                            payment_matrix[j, i] = 0

            total_payments = np.sum(payment_matrix)

            num_transactions = np.sum(payment_matrix > self.threshold)

            #double_flow_penalty = np.sum(np.minimum(payment_matrix, payment_matrix.T))

            balance_penalty = 0

            for i in range(self.n_companies):
                received = np.sum(payment_matrix[:, i])
                paid = np.sum(payment_matrix[i, :])
                net_flow = received - paid
                balance_error = abs(net_flow - self.net_positions[i])
                balance_penalty += balance_error

            fitness[p] = (payments_weight * total_payments / np.sum(self.obligations) + count_weight * num_transactions / (self.n_companies**2)
                          + balance_weight * balance_penalty / np.sum(np.abs(self.net_positions)))

        return fitness

    def optimize_with_pso(self, n_particles=50, max_iter=100):
        n_dimensions = self.n_companies * self.n_companies

        bounds = (np.zeros(n_dimensions), np.sum(self.obligations, axis=1).repeat(self.n_companies))

        #print(bounds)

        options = {
            'c1': 1.5,  # меньшее влияние личного опыта
            'c2': 2.5,  # большее влияние социального
            'w': 0.9,  # высокая инерция для широкого поиска
        }

        optimizer = ps.single.GlobalBestPSO(n_particles=n_particles,
                                            dimensions=n_dimensions,
                                            options=options,
                                            bounds=bounds)

        cost, best_solution = optimizer.optimize(self.fitness_function,
                                                 iters=max_iter,
                                                 verbose=True)

        best_payment_matrix = best_solution.reshape(self.n_companies,
                                                    self.n_companies)

        return best_payment_matrix, cost, optimizer

    def efficency_metrics(self, payment_matrix):

        total_original = np.sum(self.obligations)
        total_netting = np.sum(payment_matrix)
        reduction = (1 - total_netting / total_original) * 100

        print(f"reduction: {reduction:.2f}%")

        print("\nOptimal scheme:")
        for i in range(self.n_companies):
            for j in range(self.n_companies):
                if payment_matrix[i, j] > self.threshold:
                    print(f"Company {i} -> Company {j}: {payment_matrix[i, j]:.2f}")

        print("\nChecking balance:")
        for i in range(self.n_companies):
            net_flow = (np.sum(payment_matrix[:, i]) -
                        np.sum(payment_matrix[i, :]))
            error = abs(net_flow - self.net_positions[i])
            print(f"Company {i}: ideal sum {self.net_positions[i]:.2f}, "
                  f"fact sum {net_flow:.2f}, error {error:.2f}")


def generate_netting_matrix(n, max_value=1000):

    matrix = np.random.randint(0, max_value + 1, (n, n))
    np.fill_diagonal(matrix, 0)
    return matrix

if __name__ == "__main__":

    for i in range (3, 14):
        print("Initial matrix:")
        obligations = generate_netting_matrix(i)
        print(obligations)

        optimizer = NettingOptimizer(obligations, 0.01)

        best_payments, cost, pso_optimizer = optimizer.optimize_with_pso(
            n_particles=300, max_iter=300)

        optimizer.efficency_metrics(best_payments)

        # plt.figure(figsize=(10, 6))
        # plt.plot(pso_optimizer.cost_history)
        # plt.title('Сходимость PSO для задачи неттинга')
        # plt.xlabel('Итерация')
        # plt.ylabel('Объем платежей + штрафы')
        # plt.grid(True)
        # plt.show()