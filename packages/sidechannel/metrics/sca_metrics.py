import numpy as np
from numba import njit


class SCAMetrics:

    def __init__(self, model, traces, nt_metrics, labels_key_guess, correct_key):
        """
        :param model: object containing the (trained) keras model
        :param traces: array with (validation or attack) traces
        :param nt_metrics: number of traces to compute SCA metrics (nt_metrics <= len(traces))
        :param labels_key_guess: array with possible labels for each trace (shape: (number of key guesses, len(traces)))
        :param correct_key: integer containing the correct key value
        """
        self.model = model
        self.traces = traces
        self.nt = len(traces)
        self.nt_metrics = nt_metrics
        self.labels_key_guess = labels_key_guess
        self.correct_key = correct_key

        """ In case the user sets self.nt_metrics > self.nt"""
        if self.nt_metrics > self.nt:
            self.traces = traces[:self.nt_metrics]
            self.nt = len(self.traces)

    @njit
    def fast_key_rank(self, output_probabilities_per_key_guess, guessing_entropy_sum, success_rate_sum):
        """
        update guessing_entropy_sum and success_rate_sum
        :param output_probabilities_per_key_guess: output class probabilities per key guess
        :param guessing_entropy_sum: vector containing guessing entropy sum for each key guess
        :param success_rate_sum: vector containing success rate sum for each key guess
        :return
        """
        r = np.random.choice(self.nt, self.nt_metrics, replace=False)
        probabilities_kg_all_traces_shuffled = output_probabilities_per_key_guess[r]
        key_probabilities = np.zeros(len(probabilities_kg_all_traces_shuffled))
        kr_count = 0
        for index in range(self.nt_metrics):
            key_probabilities += probabilities_kg_all_traces_shuffled[index]
            key_probabilities_sorted = np.argsort(key_probabilities)[::-1]
            key_ranking_good_key = list(key_probabilities_sorted).index(self.correct_key) + 1
            guessing_entropy_sum[kr_count] += key_ranking_good_key
            if key_ranking_good_key == 1:
                success_rate_sum[kr_count] += 1
            kr_count += 1

    def __get_output_probabilities_per_key_guess(self, output_probabilities):
        """
        Associate labels from each key guess to respective output class probabilities
        :param output_probabilities: output class probabilities from model
        :return output_probabilities_per_key_guess: output class probabilities per key guess
        """
        nb_guesses = len(self.labels_key_guess)
        output_probabilities_per_key_guess = np.zeros((self.nt, nb_guesses))
        for index in range(self.nt):
            output_probabilities_per_key_guess[index] = output_probabilities[index][
                np.asarray([int(leakage[index]) for leakage in self.labels_key_guess[:]])
            ]
        return output_probabilities_per_key_guess

    def run(self, key_rank_executions):
        """
        :param key_rank_executions: amount of key rank executions to compose the guessing entropy and success rate
        :return:
        - vector of guessing entropy vs number of traces
        - vector of success rate vs number of traces
        """

        guessing_entropy_sum = np.zeros(self.nt_metrics)
        success_rate_sum = np.zeros(self.nt_metrics)

        output_probabilities = np.log(self.model.predict(self.traces) + 1e-36)

        output_probabilities_per_key_guess = self.__get_output_probabilities_per_key_guess(output_probabilities)

        for run in range(key_rank_executions):
            self.fast_key_rank(output_probabilities_per_key_guess, guessing_entropy_sum, success_rate_sum)

        guessing_entropy_avg = guessing_entropy_sum / key_rank_executions
        success_rate = success_rate_sum / key_rank_executions

        return guessing_entropy_avg, success_rate
