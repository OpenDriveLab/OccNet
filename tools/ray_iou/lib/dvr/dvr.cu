// Acknowledgments: https://github.com/tarashakhurana/4d-occ-forecasting
// Modified by Haisong Liu

#include <torch/extension.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>

#define MAX_D 1446 // 700 + 700 + 45 + 1
#define MAX_STEP 1000

enum LossType {L1, L2, ABSREL};
enum PhaseName {TEST, TRAIN};

template <typename scalar_t>
__global__ void init_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> points,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> tindex,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> occupancy) {

    // batch index
    const auto n = blockIdx.y;

    // ray index
    const auto c = blockIdx.x * blockDim.x + threadIdx.x;

    // num of rays
    const auto M = points.size(1);
    const auto T = occupancy.size(1);

    // we allocated more threads than num_rays
    if (c < M) {
        // ray end point
        const auto t = tindex[n][c];

        // invalid points
        assert(T == 1 || t < T);

        // if t < 0, it is a padded point
        if (t < 0) return;

        // time index for sigma
        // when T = 1, we have a static sigma
        const auto ts = (T == 1) ? 0 : t;

        // grid shape
        const int vzsize = occupancy.size(2);
        const int vysize = occupancy.size(3);
        const int vxsize = occupancy.size(4);
        // assert(vzsize + vysize + vxsize <= MAX_D);

        // end point
        const int vx = int(points[n][c][0]);
        const int vy = int(points[n][c][1]);
        const int vz = int(points[n][c][2]);

        //
        if (0 <= vx && vx < vxsize &&
            0 <= vy && vy < vysize &&
            0 <= vz && vz < vzsize) {
            occupancy[n][ts][vz][vy][vx] = 1;
        }
    }
}

template <typename scalar_t>
__global__ void render_forward_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> sigma,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> origin,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> points,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> tindex,
    // torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> pog,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> pred_dist,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> gt_dist,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> coord_index,
    PhaseName train_phase) {

    // batch index
    const auto n = blockIdx.y;

    // ray index
    const auto c = blockIdx.x * blockDim.x + threadIdx.x;

    // num of rays
    const auto M = points.size(1);
    const auto T = sigma.size(1);

    // we allocated more threads than num_rays
    if (c < M) {
        // ray end point
        const auto t = tindex[n][c];

        // invalid points
        // assert(t < T);
        assert(T == 1 || t < T);

        // time index for sigma
        // when T = 1, we have a static sigma
        const auto ts = (T == 1) ? 0 : t;

        // if t < 0, it is a padded point
        if (t < 0) return;

        // grid shape
        const int vzsize = sigma.size(2);
        const int vysize = sigma.size(3);
        const int vxsize = sigma.size(4);
        // assert(vzsize + vysize + vxsize <= MAX_D);

        // origin
        const double xo = origin[n][t][0];
        const double yo = origin[n][t][1];
        const double zo = origin[n][t][2];

        // end point
        const double xe = points[n][c][0];
        const double ye = points[n][c][1];
        const double ze = points[n][c][2];

        // locate the voxel where the origin resides
        const int vxo = int(xo);
        const int vyo = int(yo);
        const int vzo = int(zo);

        const int vxe = int(xe);
        const int vye = int(ye);
        const int vze = int(ze);

        // NOTE: new
        int vx = vxo;
        int vy = vyo;
        int vz = vzo;

        // origin to end
        const double rx = xe - xo;
        const double ry = ye - yo;
        const double rz = ze - zo;
        double gt_d = sqrt(rx * rx + ry * ry + rz * rz);

        // directional vector
        const double dx = rx / gt_d;
        const double dy = ry / gt_d;
        const double dz = rz / gt_d;

        // In which direction the voxel ids are incremented.
        const int stepX = (dx >= 0) ? 1 : -1;
        const int stepY = (dy >= 0) ? 1 : -1;
        const int stepZ = (dz >= 0) ? 1 : -1;

        // Distance along the ray to the next voxel border from the current position (tMaxX, tMaxY, tMaxZ).
        const double next_voxel_boundary_x = vx + (stepX < 0 ? 0 : 1);
        const double next_voxel_boundary_y = vy + (stepY < 0 ? 0 : 1);
        const double next_voxel_boundary_z = vz + (stepZ < 0 ? 0 : 1);

        // tMaxX, tMaxY, tMaxZ -- distance until next intersection with voxel-border
        // the value of t at which the ray crosses the first vertical voxel boundary
        double tMaxX = (dx!=0) ? (next_voxel_boundary_x - xo)/dx : DBL_MAX; //
        double tMaxY = (dy!=0) ? (next_voxel_boundary_y - yo)/dy : DBL_MAX; //
        double tMaxZ = (dz!=0) ? (next_voxel_boundary_z - zo)/dz : DBL_MAX; //

        // tDeltaX, tDeltaY, tDeltaZ --
        // how far along the ray we must move for the horizontal component to equal the width of a voxel
        // the direction in which we traverse the grid
        // can only be FLT_MAX if we never go in that direction
        const double tDeltaX = (dx!=0) ? stepX/dx : DBL_MAX;
        const double tDeltaY = (dy!=0) ? stepY/dy : DBL_MAX;
        const double tDeltaZ = (dz!=0) ? stepZ/dz : DBL_MAX;

        int3 path[MAX_D];
        double csd[MAX_D];  // cumulative sum of sigma times delta
        double p[MAX_D];  // alpha
        double d[MAX_D];

        // forward raymarching with voxel traversal
        int step = 0;  // total number of voxels traversed
        int count = 0;  // number of voxels traversed inside the voxel grid
        double last_d = 0.0;  // correct initialization

        // voxel traversal raycasting
        bool was_inside = false;
        while (true) {
            bool inside = (0 <= vx && vx < vxsize) &&
                (0 <= vy && vy < vysize) &&
                (0 <= vz && vz < vzsize);
            if (inside) {
                was_inside = true;
                path[count] = make_int3(vx, vy, vz);
            } else if (was_inside) { // was but no longer inside
                // we know we are not coming back so terminate
                break;
            } /*else if (last_d > gt_d) {
                break;
            } */
            /*else { // has not gone inside yet
                // assert(count == 0);
                // (1) when we have hit the destination but haven't gone inside the voxel grid
                // (2) when we have traveled MAX_D voxels but haven't found one valid voxel
                //     handle intersection corner cases in case of infinite loop
                bool hit = (vx == vxe && vy == vye && vz == vze);  // this test seems brittle with corner cases
                if (hit || step >= MAX_D)
                    break;
                //if (last_d >= gt_d || step >= MAX_D) break;
            } */
            // _d represents the ray distance has traveled before escaping the current voxel cell
            double _d = 0.0;
            // voxel traversal
            if (tMaxX < tMaxY) {
                if (tMaxX < tMaxZ) {
                    _d = tMaxX;
                    vx += stepX;
                    tMaxX += tDeltaX;
                } else {
                    _d = tMaxZ;
                    vz += stepZ;
                    tMaxZ += tDeltaZ;
                }
            } else {
                if (tMaxY < tMaxZ) {
                    _d = tMaxY;
                    vy += stepY;
                    tMaxY += tDeltaY;
                } else {
                    _d = tMaxZ;
                    vz += stepZ;
                    tMaxZ += tDeltaZ;
                }
            }
            if (inside) {
                // get sigma at the current voxel
                const int3 &v = path[count];  // use the recorded index
                const double _sigma = sigma[n][ts][v.z][v.y][v.x];
                const double _delta = max(0.0, _d - last_d);  // THIS TURNS OUT IMPORTANT
                const double sd = _sigma * _delta;
                if (count == 0) { // the first voxel inside
                    csd[count] = sd;
                    p[count] = 1 - exp(-sd);
                } else {
                    csd[count] = csd[count-1] + sd;
                    p[count] = exp(-csd[count-1]) - exp(-csd[count]);
                }
                // record the traveled distance
                d[count] = _d;
                // count the number of voxels we have escaped
                count ++;
            }
            last_d = _d;
            step ++;

            if (step > MAX_STEP) {
                break;
            }
        }

        // the total number of voxels visited should not exceed this number
        assert(count <= MAX_D);
        
        if (count > 0) {
            // compute the expected ray distance
            //double exp_d = 0.0;
            double exp_d = d[count-1];
            
            const int3 &v_init = path[count-1];
            int x = v_init.x;
            int y = v_init.y;
            int z = v_init.z;

            for (int i = 0; i < count; i++) {
                //printf("%f\t%f\n",p[i], d[i]);
                //exp_d += p[i] * d[i];
                const int3 &v = path[i];
                const double occ = sigma[n][ts][v.z][v.y][v.x];
                if (occ > 0.5) {
                    exp_d = d[i];
                    
                    x = v.x;
                    y = v.y;
                    z = v.z;
                
                    break;
                }

            }
            //printf("%f\n",exp_d);

            // add an imaginary sample at the end point should gt_d exceeds max_d
            double p_out = exp(-csd[count-1]);
            double max_d = d[count-1];

            // if (gt_d > max_d)
            //   exp_d += (p_out * gt_d);

            // p_out is the probability the ray escapes the voxel grid
            //exp_d += (p_out * max_d);
            if (train_phase == 1) {
                gt_d = min(gt_d, max_d);
            }

            // write the rendered ray distance (max_d)
            pred_dist[n][c] = exp_d;
            gt_dist[n][c] = gt_d;
          
            coord_index[n][c][0] = double(x);
            coord_index[n][c][1] = double(y);
            coord_index[n][c][2] = double(z);

            // // write occupancy
            // for (int i = 0; i < count; i ++) {
            //     const int3 &v = path[i];
            //     auto & occ = pog[n][t][v.z][v.y][v.x];
            //     if (p[i] >= occ) {
            //         occ = p[i];
            //     }
            // }
        }
    }
}

/*
 * input shape
 *   sigma      : N x T x H x L x W
 *   origin   : N x T x 3
 *   points   : N x M x 4
 * output shape
 *   dist     : N x M
 */
std::vector<torch::Tensor> render_forward_cuda(
    torch::Tensor sigma,
    torch::Tensor origin,
    torch::Tensor points,
    torch::Tensor tindex,
    const std::vector<int> grid,
    std::string phase_name) {

    const auto N = points.size(0); // batch size
    const auto M = points.size(1); // num of rays

    const auto T = grid[0];
    const auto H = grid[1];
    const auto L = grid[2];
    const auto W = grid[3];

    const auto device = sigma.device();

    const int threads = 1024;
    const dim3 blocks((M + threads - 1) / threads, N);

    //
    // const auto dtype = points.dtype();
    // const auto options = torch::TensorOptions().dtype(dtype).device(device).requires_grad(false);
    // auto pog = torch::zeros({N, T, H, L, W}, options);

    // perform rendering
    auto gt_dist = -torch::ones({N, M}, device);
    auto pred_dist = -torch::ones({N, M}, device);

    auto coord_index = torch::zeros({N, M, 3}, device);

    PhaseName train_phase;
    if (phase_name.compare("test") == 0) {
        train_phase = TEST;
    } else if (phase_name.compare("train") == 0){
        train_phase = TRAIN;
    } else {
        std::cout << "UNKNOWN PHASE NAME: " << phase_name << std::endl;
        exit(1);
    }

    AT_DISPATCH_FLOATING_TYPES(sigma.type(), "render_forward_cuda", ([&] {
                render_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
                    sigma.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                    origin.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    points.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    tindex.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    // pog.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                    pred_dist.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    gt_dist.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    coord_index.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    train_phase);
            }));

    cudaDeviceSynchronize();

    // return {pog, pred_dist, gt_dist};
    return {pred_dist, gt_dist, coord_index};
}

template <typename scalar_t>
__global__ void render_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> sigma,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> origin,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> points,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> tindex,
    // const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> occupancy,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> pred_dist,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> gt_dist,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_sigma,
    // torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_sigma_count,
    LossType loss_type) {

    // batch index
    const auto n = blockIdx.y;

    // ray index
    const auto c = blockIdx.x * blockDim.x + threadIdx.x;

    // num of rays
    const auto M = points.size(1);
    const auto T = sigma.size(1);

    // we allocated more threads than num_rays
    if (c < M) {
        // ray end point
        const auto t = tindex[n][c];

        // invalid points
        // assert(t < T);
        assert(T == 1 || t < T);

        // time index for sigma
        // when T = 1, we have a static sigma
        const auto ts = (T == 1) ? 0 : t;

        // if t < 0, it is a padded point
        if (t < 0) return;

        // grid shape
        const int vzsize = sigma.size(2);
        const int vysize = sigma.size(3);
        const int vxsize = sigma.size(4);
        // assert(vzsize + vysize + vxsize <= MAX_D);

        // origin
        const double xo = origin[n][t][0];
        const double yo = origin[n][t][1];
        const double zo = origin[n][t][2];

        // end point
        const double xe = points[n][c][0];
        const double ye = points[n][c][1];
        const double ze = points[n][c][2];

        // locate the voxel where the origin resides
        const int vxo = int(xo);
        const int vyo = int(yo);
        const int vzo = int(zo);

        //
        const int vxe = int(xe);
        const int vye = int(ye);
        const int vze = int(ze);

        // NOTE: new
        int vx = vxo;
        int vy = vyo;
        int vz = vzo;

        // origin to end
        const double rx = xe - xo;
        const double ry = ye - yo;
        const double rz = ze - zo;
        double gt_d = sqrt(rx * rx + ry * ry + rz * rz);

        // directional vector
        const double dx = rx / gt_d;
        const double dy = ry / gt_d;
        const double dz = rz / gt_d;

        // In which direction the voxel ids are incremented.
        const int stepX = (dx >= 0) ? 1 : -1;
        const int stepY = (dy >= 0) ? 1 : -1;
        const int stepZ = (dz >= 0) ? 1 : -1;

        // Distance along the ray to the next voxel border from the current position (tMaxX, tMaxY, tMaxZ).
        const double next_voxel_boundary_x = vx + (stepX < 0 ? 0 : 1);
        const double next_voxel_boundary_y = vy + (stepY < 0 ? 0 : 1);
        const double next_voxel_boundary_z = vz + (stepZ < 0 ? 0 : 1);

        // tMaxX, tMaxY, tMaxZ -- distance until next intersection with voxel-border
        // the value of t at which the ray crosses the first vertical voxel boundary
        double tMaxX = (dx!=0) ? (next_voxel_boundary_x - xo)/dx : DBL_MAX; //
        double tMaxY = (dy!=0) ? (next_voxel_boundary_y - yo)/dy : DBL_MAX; //
        double tMaxZ = (dz!=0) ? (next_voxel_boundary_z - zo)/dz : DBL_MAX; //

        // tDeltaX, tDeltaY, tDeltaZ --
        // how far along the ray we must move for the horizontal component to equal the width of a voxel
        // the direction in which we traverse the grid
        // can only be FLT_MAX if we never go in that direction
        const double tDeltaX = (dx!=0) ? stepX/dx : DBL_MAX;
        const double tDeltaY = (dy!=0) ? stepY/dy : DBL_MAX;
        const double tDeltaZ = (dz!=0) ? stepZ/dz : DBL_MAX;

        int3 path[MAX_D];
        double csd[MAX_D];  // cumulative sum of sigma times delta
        double p[MAX_D];  // alpha
        double d[MAX_D];
        double dt[MAX_D];

        // forward raymarching with voxel traversal
        int step = 0;  // total number of voxels traversed
        int count = 0;  // number of voxels traversed inside the voxel grid
        double last_d = 0.0;  // correct initialization

        // voxel traversal raycasting
        bool was_inside = false;
        while (true) {
            bool inside = (0 <= vx && vx < vxsize) &&
                (0 <= vy && vy < vysize) &&
                (0 <= vz && vz < vzsize);
            if (inside) { // now inside
                was_inside = true;
                path[count] = make_int3(vx, vy, vz);
            } else if (was_inside) { // was inside but no longer
                // we know we are not coming back so terminate
                break;
            } else if (last_d > gt_d) {
                break;
            } /* else { // has not gone inside yet
                // assert(count == 0);
                // (1) when we have hit the destination but haven't gone inside the voxel grid
                // (2) when we have traveled MAX_D voxels but haven't found one valid voxel
                //     handle intersection corner cases in case of infinite loop
                // bool hit = (vx == vxe && vy == vye && vz == vze);
                // if (hit || step >= MAX_D)
                //     break;
                if (last_d >= gt_d || step >= MAX_D) break;
            } */
            // _d represents the ray distance has traveled before escaping the current voxel cell
            double _d = 0.0;
            // voxel traversal
            if (tMaxX < tMaxY) {
                if (tMaxX < tMaxZ) {
                    _d = tMaxX;
                    vx += stepX;
                    tMaxX += tDeltaX;
                } else {
                    _d = tMaxZ;
                    vz += stepZ;
                    tMaxZ += tDeltaZ;
                }
            } else {
                if (tMaxY < tMaxZ) {
                    _d = tMaxY;
                    vy += stepY;
                    tMaxY += tDeltaY;
                } else {
                    _d = tMaxZ;
                    vz += stepZ;
                    tMaxZ += tDeltaZ;
                }
            }
            if (inside) {
                // get sigma at the current voxel
                const int3 &v = path[count];  // use the recorded index
                const double _sigma = sigma[n][ts][v.z][v.y][v.x];
                const double _delta = max(0.0, _d - last_d);  // THIS TURNS OUT IMPORTANT
                const double sd = _sigma * _delta;
                if (count == 0) { // the first voxel inside
                    csd[count] = sd;
                    p[count] = 1 - exp(-sd);
                } else {
                    csd[count] = csd[count-1] + sd;
                    p[count] = exp(-csd[count-1]) - exp(-csd[count]);
                }
                // record the traveled distance
                d[count] = _d;
                dt[count] = _delta;
                // count the number of voxels we have escaped
                count ++;
            }
            last_d = _d;
            step ++;

            if (step > MAX_STEP) {
                break;
            }
        }

        // the total number of voxels visited should not exceed this number
        assert(count <= MAX_D);

        // WHEN THERE IS AN INTERSECTION BETWEEN THE RAY AND THE VOXEL GRID
        if (count > 0) {
            // compute the expected ray distance
            double exp_d = 0.0;
            for (int i = 0; i < count; i ++)
                exp_d += p[i] * d[i];

            // add an imaginary sample at the end point should gt_d exceeds max_d
            double p_out = exp(-csd[count-1]);
            double max_d = d[count-1];

            exp_d += (p_out * max_d);
            gt_d = min(gt_d, max_d);

            // write the rendered ray distance (max_d)
            pred_dist[n][c] = exp_d;
            gt_dist[n][c] = gt_d;

            /* backward raymarching */
            double dd_dsigma[MAX_D];
            for (int i = count - 1; i >= 0; i --) {
                // NOTE: probably need to double check again
                if (i == count - 1)
                    dd_dsigma[i] = p_out * max_d;
                else
                    dd_dsigma[i] = dd_dsigma[i+1] - exp(-csd[i]) * (d[i+1] - d[i]);
            }

            for (int i = count - 1; i >= 0; i --)
                dd_dsigma[i] *= dt[i];

            // option 2: cap at the boundary
            for (int i = count - 1; i >= 0; i --)
                dd_dsigma[i] -= dt[i] * p_out * max_d;

            double dl_dd = 1.0;
            if (loss_type == L1)
                dl_dd = (exp_d >= gt_d) ? 1 : -1;
            else if (loss_type == L2)
                dl_dd = (exp_d - gt_d);
            else if (loss_type == ABSREL)
                dl_dd = (exp_d >= gt_d) ? (1.0/gt_d) : -(1.0/gt_d);

            // apply chain rule
            for (int i = 0; i < count; i ++) {
                const int3 &v = path[i];
                // NOTE: potential race conditions when writing gradients
                grad_sigma[n][ts][v.z][v.y][v.x] += dl_dd * dd_dsigma[i];
                // grad_sigma_count[n][ts][v.z][v.y][v.x] += 1;
            }
        }
    }
}

/*
 * input shape
 *   sigma      : N x T x H x L x W
 *   origin   : N x T x 3
 *   points   : N x M x 4
 * output shape
 *   dist     : N x M
 *   loss     : N x M
 *   grad_sigma : N x T x H x L x W
 */
std::vector<torch::Tensor> render_cuda(
    torch::Tensor sigma,
    torch::Tensor origin,
    torch::Tensor points,
    torch::Tensor tindex,
    std::string loss_name) {

    const auto N = points.size(0); // batch size
    const auto M = points.size(1); // num of rays

    const auto device = sigma.device();

    const int threads = 1024;
    const dim3 blocks((M + threads - 1) / threads, N);

    // perform rendering
    auto gt_dist = -torch::ones({N, M}, device);
    auto pred_dist = -torch::ones({N, M}, device);
    auto grad_sigma = torch::zeros_like(sigma);
    // auto grad_sigma_count = torch::zeros_like(sigma);

    LossType loss_type;
    if (loss_name.compare("l1") == 0) {
        loss_type = L1;
    } else if (loss_name.compare("l2") == 0) {
        loss_type = L2;
    } else if (loss_name.compare("absrel") == 0) {
        loss_type = ABSREL;
    } else if (loss_name.compare("bce") == 0){
        loss_type = L1;
    } else {
        std::cout << "UNKNOWN LOSS TYPE: " << loss_name << std::endl;
        exit(1);
    }

    AT_DISPATCH_FLOATING_TYPES(sigma.type(), "render_cuda", ([&] {
                render_cuda_kernel<scalar_t><<<blocks, threads>>>(
                    sigma.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                    origin.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    points.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    tindex.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    // occupancy.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                    pred_dist.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    gt_dist.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    grad_sigma.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                    // grad_sigma_count.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                    loss_type);
            }));

    cudaDeviceSynchronize();

    // grad_sigma_count += (grad_sigma_count == 0);
    // grad_sigma /= grad_sigma_count;

    return {pred_dist, gt_dist, grad_sigma};
}


/*
 * input shape
 *   origin   : N x T x 3
 *   points   : N x M x 3
 *   tindex   : N x M
 * output shape
 *   occupancy: N x T x H x L x W
 */
torch::Tensor init_cuda(
    torch::Tensor points,
    torch::Tensor tindex,
    const std::vector<int> grid) {

    const auto N = points.size(0); // batch size
    const auto M = points.size(1); // num of rays

    const auto T = grid[0];
    const auto H = grid[1];
    const auto L = grid[2];
    const auto W = grid[3];

    const auto dtype = points.dtype();
    const auto device = points.device();
    const auto options = torch::TensorOptions().dtype(dtype).device(device).requires_grad(false);
    auto occupancy = torch::zeros({N, T, H, L, W}, options);

    const int threads = 1024;
    const dim3 blocks((M + threads - 1) / threads, N);

    // initialize occupancy such that every voxel with one or more points is occupied
    AT_DISPATCH_FLOATING_TYPES(points.type(), "init_cuda", ([&] {
                init_cuda_kernel<scalar_t><<<blocks, threads>>>(
                    points.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    tindex.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    occupancy.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
            }));

    // synchronize
    cudaDeviceSynchronize();

    return occupancy;
}
