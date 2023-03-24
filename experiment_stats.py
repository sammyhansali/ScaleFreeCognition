class ExperimentStats:
    def experiment_summary(self, best_fitnesses, elapsed_times, files, nb_organisms, exp_nb, combo):
        import textwrap
        import statistics

        average_fit = sum(best_fitnesses)/nb_organisms
        average_time = sum(elapsed_times)/nb_organisms
        files.sort(key=lambda x: x.split("_")[-1], reverse=True)
        if nb_organisms>1:
            stdev=statistics.stdev(best_fitnesses)
        else:
            stdev=0

        output = f"""
        -----------------------------------------------------
                    EXPERIMENT {exp_nb} STATS

            {combo[0]}={combo[1]}

            Average Fit/Trial:      {round(average_fit,1)}
            Std. Deviation:         {round(stdev,1)}
            Max Fit:                {round(max(best_fitnesses),1)}
            Min Fit:                {round(min(best_fitnesses),1)}
            
            Average Time/Trial:     {round(average_time,1)}
            Files:                  {files}
        -----------------------------------------------------
        """
        print(textwrap.dedent(output))
        