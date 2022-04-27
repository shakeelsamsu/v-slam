#include "ceres/ceres.h"
#include "ceres/problem.h"

#include "ceres/rotation.h"

#include "stdio.h"

using namespace std;

namespace ceres
{
  struct ReprojectionError
  {
    ReprojectionError(double observed_x, double observed_y) : observed_x(observed_x), observed_y(observed_y) {}

    template <typename T>
    bool operator()(const T *const camera,
                    const T *const point,
                    T *residuals) const
    {

      // Transform from world coordinates to camera coordinates
      // P = R * X + t
      // camera[0, 1, 2] are the angle-axis rotation
      T p[3];
      AngleAxisRotatePoint(camera, point, p);
      // camera[3, 4, 5] are the translation
      p[0] += camera[3];
      p[1] += camera[4];
      p[2] += camera[5];

      // Compute the center of distortion
      // Note: the camera coordinate system has a negative z-axis
      T xp = -p[0] / p[2];
      T yp = -p[1] / p[2];

      // Apply second and fourth order radial distortion.
      // r(p) computes a scaling factor to undo the radial distortion
      // r(p) = 1.0 + k1 * ||p||^2 + k2 * ||p||^4
      const T &l1 = camera[7];
      const T &l2 = camera[8];
      T r2 = xp * xp + yp * yp;
      T distortion = 1.0 + r2 * (l1 + l2 * r2);

      // Compute final projected point position.
      // p' = f * r(p) * p
      const T &focal = camera[6];
      T predicted_x = focal * distortion * xp;
      T predicted_y = focal * distortion * yp;

      // Compute the error
      residuals[0] = predicted_x - T(observed_x);
      residuals[1] = predicted_y - T(observed_y);
      return true;
    }

    static ceres::CostFunction *Create(const double observed_x,
                                       const double observed_y)
    {
      // 2 -> number of residuals
      // 9 -> dimension of x
      // 3 -> dimension of y
      return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 9, 3>(
          new ReprojectionError(observed_x, observed_y)));
    }

    double observed_x;
    double observed_y;
  };
} // namespace ceres

class BALProblem
{
public:
  ~BALProblem()
  {
    delete[] point_index_;
    delete[] camera_index_;
    delete[] observations_;
    delete[] parameters_;
  }
  int num_observations() const { return num_observations_; }
  const double *observations() const { return observations_; }
  double *mutable_cameras() { return parameters_; }
  double *mutable_points() { return parameters_ + 9 * num_cameras_; }
  double *mutable_camera_for_observation(int i)
  {
    return mutable_cameras() + camera_index_[i] * 9;
  }
  double *mutable_point_for_observation(int i)
  {
    return mutable_points() + point_index_[i] * 3;
  }
  bool LoadFile(const char *filename)
  {
    FILE *fptr = fopen(filename, "r");
    if (fptr == nullptr)
    {
      return false;
    };
    FscanfOrDie(fptr, "%d", &num_cameras_);
    FscanfOrDie(fptr, "%d", &num_points_);
    FscanfOrDie(fptr, "%d", &num_observations_);
    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[2 * num_observations_];
    num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
    parameters_ = new double[num_parameters_];
    for (int i = 0; i < num_observations_; ++i)
    {
      FscanfOrDie(fptr, "%d", camera_index_ + i);
      FscanfOrDie(fptr, "%d", point_index_ + i);
      for (int j = 0; j < 2; ++j)
      {
        FscanfOrDie(fptr, "%lf", observations_ + 2 * i + j);
      }
    }
    for (int i = 0; i < num_parameters_; ++i)
    {
      FscanfOrDie(fptr, "%lf", parameters_ + i);
    }
    return true;
  }

private:
  template <typename T>
  void FscanfOrDie(FILE *fptr, const char *format, T *value)
  {
    int num_scanned = fscanf(fptr, format, value);
    if (num_scanned != 1)
    {
      // LOG(FATAL) << "Invalid UW data file.";
    }
  }
  int num_cameras_;
  int num_points_;
  int num_observations_;
  int num_parameters_;
  int *point_index_;
  int *camera_index_;
  double *observations_;
  double *parameters_;
};

int main()
{
  BALProblem bal_problem;
  if (!bal_problem.LoadFile("problem-16-22106-pre.txt"))
  {
    cerr << "Unable to load file." << endl;
  }

  ceres::Problem problem;
  for (int i = 0; i < bal_problem.num_observations(); ++i)
  {
    ceres::CostFunction *cost_function =
        ceres::ReprojectionError::Create(
            bal_problem.observations()[2 * i + 0],
            bal_problem.observations()[2 * i + 1]);
    problem.AddResidualBlock(cost_function,
                             nullptr /* squared loss */,
                             bal_problem.mutable_camera_for_observation(i),
                             bal_problem.mutable_point_for_observation(i));
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
  return 0;
}