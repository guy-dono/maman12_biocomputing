import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from algorithm.BulkRnaGA import BulkRnaGA


def run_with_visualization(ga: BulkRnaGA, max_generations: int = 1000, update_interval: int = 5):
    """Run GA with live heatmap visualization of population fitness."""

    plt.ion()  # Enable interactive mode

    # Collect fitness history
    fitness_history = []

    # Initial fitness
    initial_fitness = sorted([ga._fitness(ind) for ind in ga.population], reverse=True)
    fitness_history.append(initial_fitness)

    # Set up the figure
    fig, (ax_heatmap, ax_fitness, ax_rel) = plt.subplots(1, 3, figsize=(18, 6))

    # Initialize plots
    im = ax_heatmap.imshow(
        np.array(fitness_history).T,
        aspect='auto',
        cmap='RdYlGn_r',  # reversed: green=low, red=high
        interpolation='nearest',
        vmin=-30,
        vmax=-5,
    )
    cbar = plt.colorbar(im, ax=ax_heatmap, label='Fitness (negative MAE)')

    fig.show()
    fig.canvas.draw()
    fig.canvas.flush_events()

    print("Running GA...")
    while not ga.termination_condition() and ga.generation < max_generations:
        ga.step()
        gen_fitness = sorted([ga._fitness(ind) for ind in ga.population], reverse=True)
        fitness_history.append(gen_fitness)

        # Update visualization every N generations
        if ga.generation % update_interval == 0:
            best = max(gen_fitness)
            worst = min(gen_fitness)
            mean = np.mean(gen_fitness)
            rel_err = abs(best) / ga.m_mean * 100
            print(f"Gen {ga.generation}: MAE={abs(best):.2f}, rel_error={rel_err:.1f}%")

            # Update heatmap
            fitness_matrix = np.array(fitness_history)
            ax_heatmap.clear()
            im = ax_heatmap.imshow(
                fitness_matrix.T,
                aspect='auto',
                cmap='RdYlGn_r',  # reversed: green=low, red=high
                interpolation='nearest',
                vmin=-30,
                vmax=-5,
            )
            ax_heatmap.set_xlabel('Generation')
            ax_heatmap.set_ylabel('Individual (sorted by fitness)')
            ax_heatmap.set_title(f'Population Fitness Heatmap (Gen {ga.generation})')

            # Update fitness over time graph
            generations = np.arange(len(fitness_history))
            best_per_gen = [max(f) for f in fitness_history]
            mean_per_gen = [np.mean(f) for f in fitness_history]
            rel_err_per_gen = [abs(max(f)) / ga.m_mean * 100 for f in fitness_history]

            ax_fitness.clear()
            ax_fitness.plot(generations, best_per_gen, 'g-', label='Best', linewidth=2)
            ax_fitness.plot(generations, mean_per_gen, 'b-', label='Mean', linewidth=2)
            ax_fitness.set_xlabel('Generation')
            ax_fitness.set_ylabel('Fitness (negative MAE)')
            ax_fitness.set_ylim(-60, 0)
            ax_fitness.set_title('Fitness Over Time')
            ax_fitness.legend()
            ax_fitness.grid(True, alpha=0.3)

            # Update relative error graph
            ax_rel.clear()
            ax_rel.plot(generations, rel_err_per_gen, 'r-', linewidth=2)
            ax_rel.set_xlabel('Generation')
            ax_rel.set_ylabel('Relative Error (%)')
            ax_rel.set_ylim(0, 130)
            ax_rel.set_title('Relative Error Over Time')
            ax_rel.grid(True, alpha=0.3)

            fig.canvas.draw()
            fig.canvas.flush_events()

    print(f"\nFinished at generation {ga.generation}")

    plt.ioff()  # Disable interactive mode
    plt.savefig('assets/fitness_heatmap.png', dpi=150)
    print("Saved visualization to assets/fitness_heatmap.png")
    plt.show()

    return ga, fitness_history


def main():
    # Load matrices
    m_df = pd.read_csv("assets/gene_sample_TPM.tsv", sep="\t", index_col=0)
    h_df = pd.read_csv("assets/gene_celltype_TPM.tsv", sep="\t", index_col=0)

    M = m_df.values
    H = h_df.values

    print(f"M shape: {M.shape} (genes x samples)")
    print(f"H shape: {H.shape} (genes x celltypes)\n")

    ga = BulkRnaGA(
        population_size=200,
        crossover_probability=0.8,
        mutation_probability=0.3,
        m_matrix=M,
        h_matrix=H,
    )

    ga, fitness_history = run_with_visualization(ga, max_generations=1500)

    # Print final results
    best_idx = np.argmax([ga._fitness(ind) for ind in ga.population])
    best_solution = ga.population[best_idx]

    predicted = H @ best_solution
    correlation = np.corrcoef(M.flatten(), predicted.flatten())[0, 1]

    final_mae = abs(ga._fitness(best_solution))
    final_rel_error = final_mae / ga.m_mean * 100
    print(f"\nFinal MAE: {final_mae:.2f}")
    print(f"Final relative error: {final_rel_error:.2f}%")
    print(f"Correlation: {correlation:.6f}")


if __name__ == "__main__":
    main()
