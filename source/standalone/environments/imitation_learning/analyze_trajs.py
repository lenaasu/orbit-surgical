import torch, re
from pathlib import Path
import numpy as np
import pandas as pd

class AnalyzeTrajs:
    def __init__(self, traj_dir, pattern="lift_n_1_success_ep*.pt"):
        """
        Init the path and file search stratge.

        Args:
            traj_dir: the folder of trajectory files.
            pattern: the name pattern of trajectpry files.
        """
        self.traj_dir = Path(traj_dir)
        self.traj_files = sorted(self.traj_dir.glob(pattern), key=lambda p: int(re.search(r"ep(\d+)", p.stem).group(1)))

   
    def summarize(self, save_stats_path, save_summary_path):
        """Summarize the properties (length, total_reward, height) of trajectory files."""

        lengths = []
        total_rewards = []
        # final_rewards = []
        # max_heights = []
        # final_heights = [] 
        sm_state_counts = {}
        rows = []


        for traj_file in self.traj_files:
            traj = torch.load(traj_file, map_location="cpu", weights_only=True)
            rewards = [x["reward"].float().mean().item() for x in traj]
            
            # total_rewards.append(sum(rewards))
            # final_rewards.append(rewards[-1])
            # object_pos = torch.cat([x["object_pos"] for x in traj], dim=0)
            # z = object_pos[:, 2].cpu().numpy()
            # max_heights.append(z.max())
            # final_heights.append(z[-1])

            # states = torch.cat([x["sm_state"].flatten() for x in traj]).numpy()
            # for s in states:
            #     s = int(s)
            #     sm_state_counts[s] = sm_state_counts.get(s, 0) + 1
            rows.append({
                "traj_name": traj_file.stem,
                "length": len(traj),
                "total_rewards": sum(rewards),
                "mean_rewards": np.mean(rewards),
                "final_reward": rewards[-1],
                "sm_state": sm_state_counts
            })

            
        df = pd.DataFrame(rows)
        df.to_csv(save_stats_path, index=False)

        summary_df = pd.DataFrame([{
            "num_trajs": len(traj),
            "mean_len": np.mean(lengths),
            "std_len": np.std(lengths),
            "mean_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            # "mean_max_height": np.mean(max_heights),
            # "std_max_height": np.std(max_heights),
            # "mean_final_height": np.mean(final_heights),
            # "std_final_height": np.std(final_heights),
            "sm_state_counts": sm_state_counts,
        }])
        summary_df.to_csv(save_summary_path, index=False)

        # return summary





def main():    
    IL_DIR = Path(__file__).resolve().parent
    traj_dir = IL_DIR / "data" / "lift_n_trajs_50"
    save_stats_path = IL_DIR / "results" / "lift_n_50_stats.csv"
    save_summary_path = IL_DIR / "results" / "lift_n_50_summary.csv"

    # # load trajs
    # traj_files = sorted(
    #     traj_dir.glob("lift_n_1_success_ep*.pt"),
    #     key=lambda p: int(re.search(r"ep(\d+)", p.stem).group(1)),
    # )
   

    AnalyzeTrajs(traj_dir).summarize(save_stats_path=save_stats_path, save_summary_path=save_summary_path)
    print("Saved stats and summary files.")

    # print("Dataset summary:")
    # for k, v in summary.items():
    #     print(f"{k}: {v}")

    # all_obs = []
    # all_acts = []
    # all_next_obs = []
    # all_dones = []
    # all_infos = []

    # for traj_file in traj_files:
    #     traj = torch.load(traj_file, map_location="cpu", weights_only=True)

    #     for i in range(len(traj) - 1):
    #         all_obs.append(traj[i]["obs"].flatten().numpy())
    #         all_acts.append(traj[i]["action"].flatten().numpy())
    #         all_next_obs.append(traj[i + 1]["obs"].flatten().numpy())
    #         all_dones.append(bool(traj[i]["terminated"].item() or traj[i]["truncated"].item()))
    #         all_infos.append({})

    # obs = np.asarray(all_obs, dtype=np.float32)
    # acts = np.asarray(all_acts, dtype=np.float32)
    # next_obs = np.asarray(all_next_obs, dtype=np.float32)
    # dones = np.asarray(all_dones, dtype=bool)
    # infos = np.asarray(all_infos, dtype=object)

if __name__ == "__main__":
    main()