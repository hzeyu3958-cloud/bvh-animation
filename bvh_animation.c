/*
 * Copyright 1996-2022 Cyberbotics Ltd.
 *
 * Licensed under the Apache License, Version 2.0.
 *
 * Improved humanoid walking controller.
 *
 * This controller intentionally does not retarget a BVH file. It keeps the
 * Webots Skin model and drives the important bones from a procedural gait:
 * - asymmetric gait phase: 40% swing, 60% stance
 * - pelvis vertical bob and lateral weight transfer
 * - knee touchdown damping, straightening, and toe-off flex
 * - ankle heel-strike, foot-flat, and toe-off pitch
 * - arm-leg anti-phase swing with linked elbows
 * - low-pass filtering over all gait channels
 */

#include <webots/robot.h>
#include <webots/skin.h>
#include <webots/supervisor.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>

#define TIME_STEP 32
#define ARM_DOWN_ANGLE 1.42
#define WALK_SPEED 0.20
#define MAX_SKELETON_LINKS 32
#define MAX_AXIS_JOINTS 18
#define MAX_ARMOR_PARTS 52
#define AXIS_LENGTH 0.140

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
  double w;
  double x;
  double y;
  double z;
} Quaternion;

typedef struct {
  int root;
  int lower_back;
  int spine;
  int spine1;
  int neck;
  int neck1;
  int head;
  int l_shoulder;
  int l_arm;
  int l_forearm;
  int l_hand;
  int r_shoulder;
  int r_arm;
  int r_forearm;
  int r_hand;
  int l_hip;
  int l_upleg;
  int l_leg;
  int l_foot;
  int l_toe;
  int r_hip;
  int r_upleg;
  int r_leg;
  int r_foot;
  int r_toe;
} BoneIndex;

typedef struct {
  double pelvis_roll;
  double pelvis_yaw;
  double pelvis_shift_x;
  double pelvis_bob_y;
  double spine_yaw;
  double abdomen_tilt;

  double l_hip_pitch;
  double l_hip_roll;
  double l_hip_yaw;
  double l_knee;
  double l_ankle_pitch;
  double l_ankle_roll;
  double l_toe_pitch;

  double r_hip_pitch;
  double r_hip_roll;
  double r_hip_yaw;
  double r_knee;
  double r_ankle_pitch;
  double r_ankle_roll;
  double r_toe_pitch;

  double l_shoulder_pitch;
  double l_shoulder_roll;
  double l_shoulder_yaw;
  double l_elbow;
  double r_shoulder_pitch;
  double r_shoulder_roll;
  double r_shoulder_yaw;
  double r_elbow;

  double neck_yaw;
  double neck_pitch;
} GaitState;

typedef struct {
  int a;
  int b;
} BoneLink;

typedef enum {
  MECH_BOX,
  MECH_SPHERE,
  MECH_CAPSULE,
  MECH_CYLINDER
} MechShape;

typedef enum {
  MECH_WHITE,
  MECH_GRAY,
  MECH_DARK,
  MECH_BLUE
} MechMaterial;

typedef struct {
  WbFieldRef bone_points;
  WbFieldRef axis_points;
  WbFieldRef *joint_translations;
  WbFieldRef joint_scales[MAX_AXIS_JOINTS];
  WbFieldRef link_translations[MAX_SKELETON_LINKS];
  WbFieldRef link_rotations[MAX_SKELETON_LINKS];
  WbFieldRef link_heights[MAX_SKELETON_LINKS];
  WbFieldRef armor_translations[MAX_ARMOR_PARTS];
  WbFieldRef armor_rotations[MAX_ARMOR_PARTS];
  int joint_count;
  BoneLink links[MAX_SKELETON_LINKS];
  int link_count;
  int axis_bones[MAX_AXIS_JOINTS];
  int axis_count;
  int armor_bones[MAX_ARMOR_PARTS];
  double armor_offsets[MAX_ARMOR_PARTS][3];
  double armor_sizes[MAX_ARMOR_PARTS][3];
  int armor_shapes[MAX_ARMOR_PARTS];
  int armor_materials[MAX_ARMOR_PARTS];
  int armor_count;
} SkeletonOverlay;

static double clampd(double value, double lower, double upper) {
  if (value < lower)
    return lower;
  if (value > upper)
    return upper;
  return value;
}

static double smoothstep(double value) {
  value = clampd(value, 0.0, 1.0);
  return value * value * (3.0 - 2.0 * value);
}

static double lpf(double previous, double target, double alpha) {
  return previous + alpha * (target - previous);
}

static double bezier3(double t, double p0, double p1, double p2, double p3) {
  t = clampd(t, 0.0, 1.0);
  const double u = 1.0 - t;
  return u * u * u * p0 + 3.0 * u * u * t * p1 + 3.0 * u * t * t * p2 + t * t * t * p3;
}

static Quaternion quat_from_axis_angle(double x, double y, double z, double angle) {
  Quaternion q = {1.0, 0.0, 0.0, 0.0};
  const double length = sqrt(x * x + y * y + z * z);
  if (length <= 1e-10)
    return q;

  const double half_angle = 0.5 * angle;
  const double scale = sin(half_angle) / length;
  q.w = cos(half_angle);
  q.x = x * scale;
  q.y = y * scale;
  q.z = z * scale;
  return q;
}

static Quaternion quat_multiply(Quaternion a, Quaternion b) {
  Quaternion q;
  q.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
  q.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
  q.y = a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z;
  q.z = a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x;
  return q;
}

static void quat_to_axis_angle(Quaternion q, double *axis_angle) {
  const double length = sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (length <= 1e-10) {
    axis_angle[0] = 0.0;
    axis_angle[1] = 1.0;
    axis_angle[2] = 0.0;
    axis_angle[3] = 0.0;
    return;
  }

  q.w /= length;
  q.x /= length;
  q.y /= length;
  q.z /= length;

  q.w = clampd(q.w, -1.0, 1.0);
  axis_angle[3] = 2.0 * acos(q.w);
  if (axis_angle[3] < 1e-4) {
    axis_angle[0] = 0.0;
    axis_angle[1] = 1.0;
    axis_angle[2] = 0.0;
    axis_angle[3] = 0.0;
    return;
  }

  const double inv = 1.0 / sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
  axis_angle[0] = q.x * inv;
  axis_angle[1] = q.y * inv;
  axis_angle[2] = q.z * inv;
}

static void add_rot(const double *base, double x, double y, double z, double angle, double *output) {
  const Quaternion base_q = quat_from_axis_angle(base[0], base[1], base[2], base[3]);
  const Quaternion extra_q = quat_from_axis_angle(x, y, z, angle);
  quat_to_axis_angle(quat_multiply(base_q, extra_q), output);
}

static void rotate_vector_by_axis_angle(const double *axis_angle, const double *input, double *output) {
  const double axis_length =
    sqrt(axis_angle[0] * axis_angle[0] + axis_angle[1] * axis_angle[1] + axis_angle[2] * axis_angle[2]);
  if (axis_length <= 1e-10 || fabs(axis_angle[3]) <= 1e-10) {
    output[0] = input[0];
    output[1] = input[1];
    output[2] = input[2];
    return;
  }

  const double x = axis_angle[0] / axis_length;
  const double y = axis_angle[1] / axis_length;
  const double z = axis_angle[2] / axis_length;
  const double c = cos(axis_angle[3]);
  const double s = sin(axis_angle[3]);
  const double dot = x * input[0] + y * input[1] + z * input[2];

  output[0] = input[0] * c + (y * input[2] - z * input[1]) * s + x * dot * (1.0 - c);
  output[1] = input[1] * c + (z * input[0] - x * input[2]) * s + y * dot * (1.0 - c);
  output[2] = input[2] * c + (x * input[1] - y * input[0]) * s + z * dot * (1.0 - c);
}

static double vector_length(const double *v) {
  return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

static void rotation_from_y_axis_to_vector(const double *vector, double *rotation) {
  const double length = vector_length(vector);
  if (length <= 1e-8) {
    rotation[0] = 0.0;
    rotation[1] = 1.0;
    rotation[2] = 0.0;
    rotation[3] = 0.0;
    return;
  }

  const double nx = vector[0] / length;
  const double ny = vector[1] / length;
  const double nz = vector[2] / length;
  rotation[3] = acos(clampd(ny, -1.0, 1.0));

  if (rotation[3] < 1e-6) {
    rotation[0] = 0.0;
    rotation[1] = 1.0;
    rotation[2] = 0.0;
    rotation[3] = 0.0;
    return;
  }

  if (fabs(M_PI - rotation[3]) < 1e-6) {
    rotation[0] = 1.0;
    rotation[1] = 0.0;
    rotation[2] = 0.0;
    return;
  }

  rotation[0] = nz;
  rotation[1] = 0.0;
  rotation[2] = -nx;
  const double axis_length = sqrt(rotation[0] * rotation[0] + rotation[2] * rotation[2]);
  rotation[0] /= axis_length;
  rotation[2] /= axis_length;
}

static int find_bone(char **names, int count, const char *target) {
  for (int i = 0; i < count; ++i) {
    if (strcmp(names[i], target) == 0)
      return i;
  }
  return -1;
}

static BoneIndex build_bone_index(char **names, int count) {
  BoneIndex b;
  b.root = find_bone(names, count, "Hips");
  b.lower_back = find_bone(names, count, "LowerBack");
  b.spine = find_bone(names, count, "Spine");
  b.spine1 = find_bone(names, count, "Spine1");
  b.neck = find_bone(names, count, "Neck");
  b.neck1 = find_bone(names, count, "Neck1");
  b.head = find_bone(names, count, "Head");
  b.l_shoulder = find_bone(names, count, "LeftShoulder");
  b.l_arm = find_bone(names, count, "LeftArm");
  b.l_forearm = find_bone(names, count, "LeftForeArm");
  b.l_hand = find_bone(names, count, "LeftHand");
  b.r_shoulder = find_bone(names, count, "RightShoulder");
  b.r_arm = find_bone(names, count, "RightArm");
  b.r_forearm = find_bone(names, count, "RightForeArm");
  b.r_hand = find_bone(names, count, "RightHand");
  b.l_hip = find_bone(names, count, "LHipJoint");
  b.l_upleg = find_bone(names, count, "LeftUpLeg");
  b.l_leg = find_bone(names, count, "LeftLeg");
  b.l_foot = find_bone(names, count, "LeftFoot");
  b.l_toe = find_bone(names, count, "LeftToeBase");
  b.r_hip = find_bone(names, count, "RHipJoint");
  b.r_upleg = find_bone(names, count, "RightUpLeg");
  b.r_leg = find_bone(names, count, "RightLeg");
  b.r_foot = find_bone(names, count, "RightFoot");
  b.r_toe = find_bone(names, count, "RightToeBase");
  return b;
}

static bool appendf(char **cursor, size_t *remaining, const char *format, ...) {
  va_list args;
  va_start(args, format);
  const int written = vsnprintf(*cursor, *remaining, format, args);
  va_end(args);

  if (written < 0 || (size_t)written >= *remaining)
    return false;

  *cursor += written;
  *remaining -= (size_t)written;
  return true;
}

static void add_overlay_link(SkeletonOverlay *overlay, int a, int b) {
  if (a < 0 || b < 0 || overlay->link_count >= MAX_SKELETON_LINKS)
    return;

  overlay->links[overlay->link_count].a = a;
  overlay->links[overlay->link_count].b = b;
  overlay->link_count++;
}

static void add_overlay_axis(SkeletonOverlay *overlay, int bone) {
  if (bone < 0 || overlay->axis_count >= MAX_AXIS_JOINTS)
    return;

  for (int i = 0; i < overlay->axis_count; ++i) {
    if (overlay->axis_bones[i] == bone)
      return;
  }

  overlay->axis_bones[overlay->axis_count] = bone;
  overlay->axis_count++;
}

static void add_mech_part(SkeletonOverlay *overlay,
                          int bone,
                          double ox,
                          double oy,
                          double oz,
                          double sx,
                          double sy,
                          double sz,
                          MechShape shape,
                          MechMaterial material) {
  if (bone < 0 || overlay->armor_count >= MAX_ARMOR_PARTS)
    return;

  const int index = overlay->armor_count;
  overlay->armor_bones[index] = bone;
  overlay->armor_offsets[index][0] = ox;
  overlay->armor_offsets[index][1] = oy;
  overlay->armor_offsets[index][2] = oz;
  overlay->armor_sizes[index][0] = sx;
  overlay->armor_sizes[index][1] = sy;
  overlay->armor_sizes[index][2] = sz;
  overlay->armor_shapes[index] = shape;
  overlay->armor_materials[index] = material;
  overlay->armor_count++;
}

static void add_armor_part(SkeletonOverlay *overlay,
                           int bone,
                           double ox,
                           double oy,
                           double oz,
                           double sx,
                           double sy,
                           double sz) {
  add_mech_part(overlay, bone, ox, oy, oz, sx, sy, sz, MECH_BOX, MECH_WHITE);
}

static void setup_overlay_topology(SkeletonOverlay *overlay, const BoneIndex *bones) {
  add_overlay_link(overlay, bones->root, bones->lower_back);
  add_overlay_link(overlay, bones->lower_back, bones->spine);
  add_overlay_link(overlay, bones->spine, bones->spine1);
  add_overlay_link(overlay, bones->spine1, bones->neck);
  add_overlay_link(overlay, bones->neck, bones->neck1);
  add_overlay_link(overlay, bones->neck1, bones->head);

  add_overlay_link(overlay, bones->spine1, bones->l_shoulder);
  add_overlay_link(overlay, bones->l_shoulder, bones->l_arm);
  add_overlay_link(overlay, bones->l_arm, bones->l_forearm);
  add_overlay_link(overlay, bones->l_forearm, bones->l_hand);
  add_overlay_link(overlay, bones->spine1, bones->r_shoulder);
  add_overlay_link(overlay, bones->r_shoulder, bones->r_arm);
  add_overlay_link(overlay, bones->r_arm, bones->r_forearm);
  add_overlay_link(overlay, bones->r_forearm, bones->r_hand);

  add_overlay_link(overlay, bones->root, bones->l_hip);
  add_overlay_link(overlay, bones->l_hip, bones->l_upleg);
  add_overlay_link(overlay, bones->l_upleg, bones->l_leg);
  add_overlay_link(overlay, bones->l_leg, bones->l_foot);
  add_overlay_link(overlay, bones->l_foot, bones->l_toe);
  add_overlay_link(overlay, bones->root, bones->r_hip);
  add_overlay_link(overlay, bones->r_hip, bones->r_upleg);
  add_overlay_link(overlay, bones->r_upleg, bones->r_leg);
  add_overlay_link(overlay, bones->r_leg, bones->r_foot);
  add_overlay_link(overlay, bones->r_foot, bones->r_toe);

  add_overlay_axis(overlay, bones->root);
  add_overlay_axis(overlay, bones->lower_back);
  add_overlay_axis(overlay, bones->spine1);
  add_overlay_axis(overlay, bones->neck);
  add_overlay_axis(overlay, bones->l_shoulder);
  add_overlay_axis(overlay, bones->l_arm);
  add_overlay_axis(overlay, bones->l_forearm);
  add_overlay_axis(overlay, bones->r_shoulder);
  add_overlay_axis(overlay, bones->r_arm);
  add_overlay_axis(overlay, bones->r_forearm);
  add_overlay_axis(overlay, bones->l_hip);
  add_overlay_axis(overlay, bones->l_upleg);
  add_overlay_axis(overlay, bones->l_leg);
  add_overlay_axis(overlay, bones->l_foot);
  add_overlay_axis(overlay, bones->r_hip);
  add_overlay_axis(overlay, bones->r_upleg);
  add_overlay_axis(overlay, bones->r_leg);
  add_overlay_axis(overlay, bones->r_foot);

  add_mech_part(overlay, bones->head, 0.0, 0.09, 0.0, 0.090, 0.180, 0.0, MECH_CAPSULE, MECH_WHITE);
  add_mech_part(overlay, bones->head, 0.0, 0.11, 0.070, 0.110, 0.035, 0.018, MECH_BOX, MECH_GRAY);
  add_mech_part(overlay, bones->neck, 0.0, 0.02, 0.0, 0.045, 0.075, 0.0, MECH_CYLINDER, MECH_DARK);

  add_armor_part(overlay, bones->spine1, 0.0, -0.06, 0.035, 0.46, 0.34, 0.16);
  add_mech_part(overlay, bones->spine1, 0.0, -0.09, 0.120, 0.085, 0.0, 0.0, MECH_SPHERE, MECH_GRAY);
  add_mech_part(overlay, bones->spine1, -0.19, -0.03, 0.02, 0.10, 0.28, 0.16, MECH_BOX, MECH_WHITE);
  add_mech_part(overlay, bones->spine1, 0.19, -0.03, 0.02, 0.10, 0.28, 0.16, MECH_BOX, MECH_WHITE);
  add_mech_part(overlay, bones->spine, 0.0, -0.03, 0.025, 0.30, 0.24, 0.13, MECH_BOX, MECH_WHITE);
  add_mech_part(overlay, bones->lower_back, 0.0, -0.03, 0.025, 0.25, 0.16, 0.12, MECH_BOX, MECH_GRAY);
  add_mech_part(overlay, bones->root, 0.0, -0.04, 0.0, 0.36, 0.16, 0.18, MECH_BOX, MECH_WHITE);
  add_mech_part(overlay, bones->root, -0.13, -0.08, 0.065, 0.12, 0.08, 0.10, MECH_BOX, MECH_WHITE);
  add_mech_part(overlay, bones->root, 0.13, -0.08, 0.065, 0.12, 0.08, 0.10, MECH_BOX, MECH_WHITE);

  add_mech_part(overlay, bones->l_shoulder, 0.0, -0.02, 0.02, 0.16, 0.10, 0.16, MECH_BOX, MECH_WHITE);
  add_mech_part(overlay, bones->r_shoulder, 0.0, -0.02, 0.02, 0.16, 0.10, 0.16, MECH_BOX, MECH_WHITE);
  add_mech_part(overlay, bones->l_arm, 0.0, -0.11, 0.0, 0.13, 0.24, 0.11, MECH_BOX, MECH_WHITE);
  add_mech_part(overlay, bones->r_arm, 0.0, -0.11, 0.0, 0.13, 0.24, 0.11, MECH_BOX, MECH_WHITE);
  add_mech_part(overlay, bones->l_forearm, 0.0, -0.14, 0.0, 0.10, 0.28, 0.10, MECH_BOX, MECH_WHITE);
  add_mech_part(overlay, bones->r_forearm, 0.0, -0.14, 0.0, 0.10, 0.28, 0.10, MECH_BOX, MECH_WHITE);
  add_mech_part(overlay, bones->l_hand, 0.0, -0.04, 0.0, 0.055, 0.085, 0.035, MECH_BOX, MECH_DARK);
  add_mech_part(overlay, bones->r_hand, 0.0, -0.04, 0.0, 0.055, 0.085, 0.035, MECH_BOX, MECH_DARK);

  add_mech_part(overlay, bones->l_upleg, 0.0, -0.18, 0.0, 0.15, 0.35, 0.13, MECH_BOX, MECH_WHITE);
  add_mech_part(overlay, bones->r_upleg, 0.0, -0.18, 0.0, 0.15, 0.35, 0.13, MECH_BOX, MECH_WHITE);
  add_mech_part(overlay, bones->l_upleg, 0.065, -0.20, 0.0, 0.045, 0.29, 0.0, MECH_CYLINDER, MECH_GRAY);
  add_mech_part(overlay, bones->r_upleg, -0.065, -0.20, 0.0, 0.045, 0.29, 0.0, MECH_CYLINDER, MECH_GRAY);
  add_mech_part(overlay, bones->l_leg, 0.0, -0.19, 0.0, 0.12, 0.37, 0.11, MECH_BOX, MECH_DARK);
  add_mech_part(overlay, bones->r_leg, 0.0, -0.19, 0.0, 0.12, 0.37, 0.11, MECH_BOX, MECH_DARK);
  add_mech_part(overlay, bones->l_leg, 0.0, -0.10, 0.065, 0.10, 0.18, 0.06, MECH_BOX, MECH_WHITE);
  add_mech_part(overlay, bones->r_leg, 0.0, -0.10, 0.065, 0.10, 0.18, 0.06, MECH_BOX, MECH_WHITE);
  add_mech_part(overlay, bones->l_foot, 0.0, -0.03, 0.08, 0.13, 0.08, 0.28, MECH_BOX, MECH_DARK);
  add_mech_part(overlay, bones->r_foot, 0.0, -0.03, 0.08, 0.13, 0.08, 0.28, MECH_BOX, MECH_DARK);
  add_mech_part(overlay, bones->l_toe, 0.0, 0.0, 0.08, 0.11, 0.045, 0.16, MECH_BOX, MECH_DARK);
  add_mech_part(overlay, bones->r_toe, 0.0, 0.0, 0.08, 0.11, 0.045, 0.16, MECH_BOX, MECH_DARK);
}

static bool append_mech_material(char **cursor, size_t *remaining, MechMaterial material) {
  double diffuse[3] = {0.82, 0.84, 0.84};
  double emissive[3] = {0.0, 0.0, 0.0};
  double specular[3] = {0.85, 0.85, 0.85};

  if (material == MECH_GRAY) {
    diffuse[0] = 0.50;
    diffuse[1] = 0.52;
    diffuse[2] = 0.53;
    specular[0] = 0.60;
    specular[1] = 0.62;
    specular[2] = 0.64;
  } else if (material == MECH_DARK) {
    diffuse[0] = 0.018;
    diffuse[1] = 0.018;
    diffuse[2] = 0.022;
    specular[0] = 0.35;
    specular[1] = 0.35;
    specular[2] = 0.38;
  } else if (material == MECH_BLUE) {
    diffuse[0] = 0.05;
    diffuse[1] = 0.16;
    diffuse[2] = 0.85;
    emissive[0] = 0.02;
    emissive[1] = 0.04;
    emissive[2] = 0.28;
    specular[0] = 0.25;
    specular[1] = 0.35;
    specular[2] = 1.0;
  }

  return appendf(cursor,
                 remaining,
                 "appearance Appearance { material Material { diffuseColor %.3f %.3f %.3f emissiveColor %.3f %.3f %.3f "
                 "specularColor %.3f %.3f %.3f } }",
                 diffuse[0],
                 diffuse[1],
                 diffuse[2],
                 emissive[0],
                 emissive[1],
                 emissive[2],
                 specular[0],
                 specular[1],
                 specular[2]);
}

static bool append_mech_geometry(char **cursor, size_t *remaining, int index, MechShape shape, double sx, double sy, double sz) {
  if (shape == MECH_SPHERE)
    return appendf(cursor, remaining, "geometry Sphere { radius %.4f }", sx);
  if (shape == MECH_CAPSULE)
    return appendf(cursor, remaining, "geometry DEF GAIT_MECH_ARMOR_CAPSULE_%d Capsule { radius %.4f height %.4f }", index, sx, sy);
  if (shape == MECH_CYLINDER)
    return appendf(cursor, remaining, "geometry Cylinder { radius %.4f height %.4f }", sx, sy);
  return appendf(cursor, remaining, "geometry Box { size %.4f %.4f %.4f }", sx, sy, sz);
}

static bool create_skeleton_overlay(SkeletonOverlay *overlay, const BoneIndex *bones, int bone_count) {
  memset(overlay, 0, sizeof(SkeletonOverlay));
  overlay->joint_count = bone_count;
  overlay->joint_translations = (WbFieldRef *)calloc((size_t)bone_count, sizeof(WbFieldRef));
  if (!overlay->joint_translations)
    return false;

  setup_overlay_topology(overlay, bones);

  WbNodeRef existing_overlay = wb_supervisor_node_get_from_def("GAIT_DEBUG_OVERLAY");
  if (existing_overlay)
    wb_supervisor_node_remove(existing_overlay);

  const size_t capacity = 32768 + (size_t)bone_count * 560 + MAX_SKELETON_LINKS * 520 + MAX_AXIS_JOINTS * 280 +
                          MAX_ARMOR_PARTS * 520;
  char *node_string = (char *)malloc(capacity);
  if (!node_string) {
    free(overlay->joint_translations);
    overlay->joint_translations = NULL;
    return false;
  }

  char *cursor = node_string;
  size_t remaining = capacity;
  bool ok = true;

  ok = ok && appendf(&cursor,
                     &remaining,
                     "DEF GAIT_DEBUG_OVERLAY Transform {\n"
                     "  rotation 0.57735 0.57735 0.57735 2.0944\n"
                     "  children [\n");
  for (int i = 0; i < MAX_ARMOR_PARTS; ++i) {
    const double sx = i < overlay->armor_count ? overlay->armor_sizes[i][0] : 0.01;
    const double sy = i < overlay->armor_count ? overlay->armor_sizes[i][1] : 0.01;
    const double sz = i < overlay->armor_count ? overlay->armor_sizes[i][2] : 0.01;
    const MechShape shape = i < overlay->armor_count ? (MechShape)overlay->armor_shapes[i] : MECH_BOX;
    const MechMaterial material = i < overlay->armor_count ? (MechMaterial)overlay->armor_materials[i] : MECH_WHITE;
    ok = ok && appendf(&cursor,
                       &remaining,
                       "    DEF GAIT_MECH_ARMOR_%d Transform {\n"
                       "      translation 0 0 0\n"
                       "      rotation 0 1 0 0\n"
                       "      children [ Shape { ",
                       i);
    ok = ok && append_mech_material(&cursor, &remaining, material);
    ok = ok && appendf(&cursor, &remaining, " ");
    ok = ok && append_mech_geometry(&cursor, &remaining, i, shape, sx, sy, sz);
    ok = ok && appendf(&cursor, &remaining, " } ]\n    }\n");
  }

  for (int i = 0; i < MAX_SKELETON_LINKS; ++i) {
    ok = ok && appendf(&cursor,
                       &remaining,
                       "    DEF GAIT_MECH_BONE_%d Transform {\n"
                       "      translation 0 0 0\n"
                       "      rotation 0 1 0 0\n"
                       "      children [ Shape { appearance Appearance { material Material { diffuseColor 0.05 0.16 0.85 "
                       "emissiveColor 0.02 0.04 0.28 specularColor 0.25 0.35 1 } } geometry DEF GAIT_MECH_BONE_CAPSULE_%d "
                       "Capsule { radius 0.018 height 0.100 } } ]\n"
                       "    }\n",
                       i,
                       i);
  }

  ok = ok && appendf(&cursor,
                     &remaining,
                     "    Shape {\n"
                     "      appearance Appearance { material Material { diffuseColor 0.05 0.75 1 "
                     "emissiveColor 0.02 0.40 0.75 transparency 0.05 } }\n"
                     "      geometry IndexedLineSet {\n"
                     "        coord DEF GAIT_DEBUG_BONE_COORD Coordinate { point [\n");
  for (int i = 0; i < MAX_SKELETON_LINKS * 2; ++i)
    ok = ok && appendf(&cursor, &remaining, "          0 0 0\n");
  ok = ok && appendf(&cursor, &remaining, "        ] }\n        coordIndex [\n");
  for (int i = 0; i < MAX_SKELETON_LINKS; ++i)
    ok = ok && appendf(&cursor, &remaining, "          %d %d -1\n", i * 2, i * 2 + 1);
  ok = ok && appendf(&cursor, &remaining, "        ]\n      }\n    }\n");

  ok = ok && appendf(&cursor,
                     &remaining,
                     "    Shape {\n"
                     "      geometry IndexedLineSet {\n"
                     "        color Color { color [ 1 0.04 0.04, 0.10 1 0.18, 0.12 0.28 1 ] }\n"
                     "        colorPerVertex FALSE\n"
                     "        coord DEF GAIT_DEBUG_AXIS_COORD Coordinate { point [\n");
  for (int i = 0; i < MAX_AXIS_JOINTS * 6; ++i)
    ok = ok && appendf(&cursor, &remaining, "          0 0 0\n");
  ok = ok && appendf(&cursor, &remaining, "        ] }\n        coordIndex [\n");
  for (int i = 0; i < MAX_AXIS_JOINTS * 3; ++i)
    ok = ok && appendf(&cursor, &remaining, "          %d %d -1\n", i * 2, i * 2 + 1);
  ok = ok && appendf(&cursor, &remaining, "        ]\n        colorIndex [\n");
  for (int i = 0; i < MAX_AXIS_JOINTS; ++i)
    ok = ok && appendf(&cursor, &remaining, "          0 1 2\n");
  ok = ok && appendf(&cursor, &remaining, "        ]\n      }\n    }\n");

  for (int i = 0; i < bone_count; ++i) {
    ok = ok && appendf(&cursor,
                       &remaining,
                       "    DEF GAIT_DEBUG_JOINT_%d Transform {\n"
                       "      translation 0 0 0\n"
                       "      children [ Shape { appearance Appearance { material Material { diffuseColor 0.015 0.015 0.018 "
                       "specularColor 0.35 0.35 0.38 } } geometry Sphere { radius 0.030 } } ]\n"
                       "    }\n",
                       i);
  }

  ok = ok && appendf(&cursor, &remaining, "  ]\n}\n");

  if (!ok) {
    fprintf(stderr, "Warning: failed to build the visible skeleton overlay string.\n");
    free(node_string);
    free(overlay->joint_translations);
    overlay->joint_translations = NULL;
    return false;
  }

  WbNodeRef self_node = wb_supervisor_node_get_self();
  WbFieldRef self_children = wb_supervisor_node_get_field(self_node, "children");
  wb_supervisor_field_import_mf_node_from_string(self_children, -1, node_string);
  free(node_string);

  WbNodeRef bone_coord = wb_supervisor_node_get_from_def("GAIT_DEBUG_BONE_COORD");
  WbNodeRef axis_coord = wb_supervisor_node_get_from_def("GAIT_DEBUG_AXIS_COORD");
  overlay->bone_points = bone_coord ? wb_supervisor_node_get_field(bone_coord, "point") : 0;
  overlay->axis_points = axis_coord ? wb_supervisor_node_get_field(axis_coord, "point") : 0;

  for (int i = 0; i < bone_count; ++i) {
    char def_name[64];
    snprintf(def_name, sizeof(def_name), "GAIT_DEBUG_JOINT_%d", i);
    WbNodeRef joint_node = wb_supervisor_node_get_from_def(def_name);
    overlay->joint_translations[i] = joint_node ? wb_supervisor_node_get_field(joint_node, "translation") : 0;
  }

  for (int i = 0; i < MAX_SKELETON_LINKS; ++i) {
    char def_name[80];
    snprintf(def_name, sizeof(def_name), "GAIT_MECH_BONE_%d", i);
    WbNodeRef link_node = wb_supervisor_node_get_from_def(def_name);
    overlay->link_translations[i] = link_node ? wb_supervisor_node_get_field(link_node, "translation") : 0;
    overlay->link_rotations[i] = link_node ? wb_supervisor_node_get_field(link_node, "rotation") : 0;

    snprintf(def_name, sizeof(def_name), "GAIT_MECH_BONE_CAPSULE_%d", i);
    WbNodeRef capsule_node = wb_supervisor_node_get_from_def(def_name);
    overlay->link_heights[i] = capsule_node ? wb_supervisor_node_get_field(capsule_node, "height") : 0;
  }

  for (int i = 0; i < MAX_ARMOR_PARTS; ++i) {
    char def_name[80];
    snprintf(def_name, sizeof(def_name), "GAIT_MECH_ARMOR_%d", i);
    WbNodeRef armor_node = wb_supervisor_node_get_from_def(def_name);
    overlay->armor_translations[i] = armor_node ? wb_supervisor_node_get_field(armor_node, "translation") : 0;
    overlay->armor_rotations[i] = armor_node ? wb_supervisor_node_get_field(armor_node, "rotation") : 0;
  }

  if (!overlay->bone_points || !overlay->axis_points) {
    fprintf(stderr, "Warning: visible skeleton overlay was created but could not be wired for live updates.\n");
    return false;
  }

  return true;
}

static void update_skeleton_overlay(WbDeviceTag skin, const SkeletonOverlay *overlay) {
  if (!overlay || !overlay->joint_translations)
    return;

  double zero[3] = {0.0, 0.0, 0.0};
  for (int i = 0; i < overlay->joint_count; ++i) {
    const double *position = wb_skin_get_bone_position(skin, i, true);
    if (overlay->joint_translations[i])
      wb_supervisor_field_set_sf_vec3f(overlay->joint_translations[i], position ? position : zero);
  }

  if (overlay->bone_points) {
    for (int i = 0; i < MAX_SKELETON_LINKS; ++i) {
      const double *a = zero;
      const double *b = zero;
      if (i < overlay->link_count) {
        const double *pa = wb_skin_get_bone_position(skin, overlay->links[i].a, true);
        const double *pb = wb_skin_get_bone_position(skin, overlay->links[i].b, true);
        if (pa && pb) {
          a = pa;
          b = pb;
        }
      }
      wb_supervisor_field_set_mf_vec3f(overlay->bone_points, i * 2, a);
      wb_supervisor_field_set_mf_vec3f(overlay->bone_points, i * 2 + 1, b);

      if (overlay->link_translations[i] && overlay->link_rotations[i] && overlay->link_heights[i]) {
        const double vector[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const double length = i < overlay->link_count ? vector_length(vector) : 0.001;
        double midpoint[3] = {
          0.5 * (a[0] + b[0]),
          0.5 * (a[1] + b[1]),
          0.5 * (a[2] + b[2])
        };
        double rotation[4];
        rotation_from_y_axis_to_vector(vector, rotation);
        wb_supervisor_field_set_sf_vec3f(overlay->link_translations[i], midpoint);
        wb_supervisor_field_set_sf_rotation(overlay->link_rotations[i], rotation);
        wb_supervisor_field_set_sf_float(overlay->link_heights[i], fmax(length - 0.036, 0.001));
      }
    }
  }

  for (int i = 0; i < MAX_ARMOR_PARTS; ++i) {
    const int bone = i < overlay->armor_count ? overlay->armor_bones[i] : -1;
    const double *center = bone >= 0 ? wb_skin_get_bone_position(skin, bone, true) : zero;
    const double *orientation = bone >= 0 ? wb_skin_get_bone_orientation(skin, bone, true) : NULL;
    if (!center)
      center = zero;

    double translation[3] = {center[0], center[1], center[2]};
    double rotated_offset[3] = {0.0, 0.0, 0.0};
    if (bone >= 0 && orientation)
      rotate_vector_by_axis_angle(orientation, overlay->armor_offsets[i], rotated_offset);
    translation[0] += rotated_offset[0];
    translation[1] += rotated_offset[1];
    translation[2] += rotated_offset[2];

    if (overlay->armor_translations[i])
      wb_supervisor_field_set_sf_vec3f(overlay->armor_translations[i], translation);
    if (overlay->armor_rotations[i] && orientation)
      wb_supervisor_field_set_sf_rotation(overlay->armor_rotations[i], orientation);
  }

  if (overlay->axis_points) {
    const double local_axes[3][3] = {
      {AXIS_LENGTH, 0.0, 0.0},
      {0.0, AXIS_LENGTH, 0.0},
      {0.0, 0.0, AXIS_LENGTH}
    };

    for (int i = 0; i < MAX_AXIS_JOINTS; ++i) {
      const int bone = i < overlay->axis_count ? overlay->axis_bones[i] : -1;
      const double *center = bone >= 0 ? wb_skin_get_bone_position(skin, bone, true) : zero;
      const double *orientation = bone >= 0 ? wb_skin_get_bone_orientation(skin, bone, true) : NULL;
      if (!center)
        center = zero;

      for (int axis = 0; axis < 3; ++axis) {
        double end[3] = {center[0], center[1], center[2]};
        double rotated[3] = {local_axes[axis][0], local_axes[axis][1], local_axes[axis][2]};
        if (orientation)
          rotate_vector_by_axis_angle(orientation, local_axes[axis], rotated);
        end[0] += rotated[0];
        end[1] += rotated[1];
        end[2] += rotated[2];

        const int base = i * 6 + axis * 2;
        wb_supervisor_field_set_mf_vec3f(overlay->axis_points, base, center);
        wb_supervisor_field_set_mf_vec3f(overlay->axis_points, base + 1, end);
      }
    }
  }
}

static void cleanup_skeleton_overlay(SkeletonOverlay *overlay) {
  free(overlay->joint_translations);
  memset(overlay, 0, sizeof(SkeletonOverlay));
}

static void print_usage(const char *command) {
  printf("Usage: %s -d <skin_device_name> [-g <gait_frequency>]\n", command);
  printf("Compatibility: -f, -s, -e, and -l are accepted and ignored.\n");
}

static void compute_gait_target(double cycle, GaitState *target) {
  const double swing_ratio = 0.40;
  const double left_cycle = cycle;
  const double right_cycle = fmod(cycle + 0.5, 1.0);
  const double walk_phase = sin(2.0 * M_PI * cycle);
  const double support_phase = cos(2.0 * M_PI * cycle);
  const double weight_phase = sin(4.0 * M_PI * cycle);
  const double arm_phase = 2.0 * M_PI * cycle;

  const bool left_swing = left_cycle < swing_ratio;
  const bool right_swing = right_cycle < swing_ratio;
  const double left_swing_t = left_swing ? left_cycle / swing_ratio : 0.0;
  const double right_swing_t = right_swing ? right_cycle / swing_ratio : 0.0;
  const double left_stance_t = left_swing ? 0.0 : (left_cycle - swing_ratio) / (1.0 - swing_ratio);
  const double right_stance_t = right_swing ? 0.0 : (right_cycle - swing_ratio) / (1.0 - swing_ratio);

  target->pelvis_shift_x = 0.026 * support_phase;
  target->pelvis_bob_y = -0.016 * (0.35 + 0.65 * fabs(weight_phase));
  target->pelvis_roll = 0.092 * support_phase;
  target->pelvis_yaw = 0.125 * walk_phase;
  target->abdomen_tilt = 0.036 + 0.018 * fabs(walk_phase);
  target->spine_yaw = -0.110 * walk_phase;

  if (left_swing) {
    const double t = left_swing_t;
    target->l_hip_pitch = bezier3(t, -0.12, 0.03, 0.31, 0.18);
    target->l_hip_roll = 0.034 * sin(M_PI * t);
    target->l_hip_yaw = 0.024 * sin(M_PI * t);
    target->l_knee = bezier3(t, 0.10, 0.55, 0.61, 0.15);
    target->l_ankle_pitch = bezier3(t, 0.04, 0.15, 0.06, -0.09);
    target->l_ankle_roll = -0.050 * sin(M_PI * t);
    target->l_toe_pitch = 0.05 * smoothstep(t);
  } else {
    const double t = left_stance_t;
    const double toe_off = smoothstep((t - 0.65) / 0.35);
    const double heel_strike = 1.0 - smoothstep(t / 0.25);
    target->l_hip_pitch = bezier3(t, 0.18, 0.10, -0.11, -0.16);
    target->l_hip_roll = -0.030 * (1.0 - t);
    target->l_hip_yaw = -0.020 * t;
    target->l_knee = 0.050 + 0.11 * smoothstep(1.0 - clampd(t * 3.2, 0.0, 1.0)) + 0.055 * toe_off;
    target->l_ankle_pitch = -0.115 * heel_strike + 0.20 * toe_off;
    target->l_ankle_roll = 0.066 * (1.0 - t);
    target->l_toe_pitch = 0.055 * toe_off;
  }

  if (right_swing) {
    const double t = right_swing_t;
    target->r_hip_pitch = bezier3(t, -0.12, 0.03, 0.31, 0.18);
    target->r_hip_roll = -0.034 * sin(M_PI * t);
    target->r_hip_yaw = -0.024 * sin(M_PI * t);
    target->r_knee = bezier3(t, 0.10, 0.55, 0.61, 0.15);
    target->r_ankle_pitch = bezier3(t, 0.04, 0.15, 0.06, -0.09);
    target->r_ankle_roll = 0.050 * sin(M_PI * t);
    target->r_toe_pitch = 0.05 * smoothstep(t);
  } else {
    const double t = right_stance_t;
    const double toe_off = smoothstep((t - 0.65) / 0.35);
    const double heel_strike = 1.0 - smoothstep(t / 0.25);
    target->r_hip_pitch = bezier3(t, 0.18, 0.10, -0.11, -0.16);
    target->r_hip_roll = 0.030 * (1.0 - t);
    target->r_hip_yaw = 0.020 * t;
    target->r_knee = 0.050 + 0.11 * smoothstep(1.0 - clampd(t * 3.2, 0.0, 1.0)) + 0.055 * toe_off;
    target->r_ankle_pitch = -0.115 * heel_strike + 0.20 * toe_off;
    target->r_ankle_roll = -0.066 * (1.0 - t);
    target->r_toe_pitch = 0.055 * toe_off;
  }

  target->l_shoulder_pitch = 0.31 * sin(arm_phase + M_PI) - 0.025 * support_phase;
  target->r_shoulder_pitch = 0.31 * sin(arm_phase) + 0.025 * support_phase;
  target->l_shoulder_roll = -0.075 + 0.016 * sin(arm_phase + 0.35);
  target->r_shoulder_roll = 0.075 + 0.016 * sin(arm_phase + M_PI + 0.35);
  target->l_shoulder_yaw = -0.050 * sin(arm_phase + M_PI + 0.55);
  target->r_shoulder_yaw = 0.050 * sin(arm_phase + 0.55);
  target->l_elbow = 0.26 + 0.13 * (0.5 + 0.5 * sin(arm_phase + M_PI + 0.75));
  target->r_elbow = 0.26 + 0.13 * (0.5 + 0.5 * sin(arm_phase + 0.75));

  target->neck_yaw = -0.060 * walk_phase;
  target->neck_pitch = 0.012 * support_phase;
}

static void lpf_state(GaitState *state, const GaitState *target, double alpha) {
#define LPF_FIELD(field) state->field = lpf(state->field, target->field, alpha)
  LPF_FIELD(pelvis_roll);
  LPF_FIELD(pelvis_yaw);
  LPF_FIELD(pelvis_shift_x);
  LPF_FIELD(pelvis_bob_y);
  LPF_FIELD(spine_yaw);
  LPF_FIELD(abdomen_tilt);
  LPF_FIELD(l_hip_pitch);
  LPF_FIELD(l_hip_roll);
  LPF_FIELD(l_hip_yaw);
  LPF_FIELD(l_knee);
  LPF_FIELD(l_ankle_pitch);
  LPF_FIELD(l_ankle_roll);
  LPF_FIELD(l_toe_pitch);
  LPF_FIELD(r_hip_pitch);
  LPF_FIELD(r_hip_roll);
  LPF_FIELD(r_hip_yaw);
  LPF_FIELD(r_knee);
  LPF_FIELD(r_ankle_pitch);
  LPF_FIELD(r_ankle_roll);
  LPF_FIELD(r_toe_pitch);
  LPF_FIELD(l_shoulder_pitch);
  LPF_FIELD(l_shoulder_roll);
  LPF_FIELD(l_shoulder_yaw);
  LPF_FIELD(l_elbow);
  LPF_FIELD(r_shoulder_pitch);
  LPF_FIELD(r_shoulder_roll);
  LPF_FIELD(r_shoulder_yaw);
  LPF_FIELD(r_elbow);
  LPF_FIELD(neck_yaw);
  LPF_FIELD(neck_pitch);
#undef LPF_FIELD
}

static void apply_gait(WbDeviceTag skin,
                       const BoneIndex *bones,
                       int bone_count,
                       const double *neutral,
                       const GaitState *gait,
                       double cycle) {
  const double hand_pitch = 0.02 * sin(2.0 * M_PI * cycle + 0.4);

  for (int i = 0; i < bone_count; ++i) {
    double orientation[4] = {neutral[i * 4], neutral[i * 4 + 1], neutral[i * 4 + 2], neutral[i * 4 + 3]};

    if (i == bones->root) {
      add_rot(orientation, 1.0, 0.0, 0.0, gait->abdomen_tilt, orientation);
      add_rot(orientation, 0.0, 1.0, 0.0, gait->pelvis_roll, orientation);
      add_rot(orientation, 0.0, 0.0, 1.0, gait->pelvis_yaw * 0.90, orientation);
    } else if (i == bones->lower_back || i == bones->spine) {
      add_rot(orientation, 1.0, 0.0, 0.0, gait->abdomen_tilt * 0.45, orientation);
      add_rot(orientation, 0.0, 1.0, 0.0, gait->pelvis_roll * 0.65, orientation);
      add_rot(orientation, 0.0, 0.0, 1.0, -gait->pelvis_yaw * 0.55, orientation);
    } else if (i == bones->spine1) {
      add_rot(orientation, 1.0, 0.0, 0.0, -gait->abdomen_tilt * 0.30, orientation);
      add_rot(orientation, 0.0, 1.0, 0.0, -gait->pelvis_roll * 0.30, orientation);
      add_rot(orientation, 0.0, 0.0, 1.0, gait->spine_yaw * 1.45, orientation);
    } else if (i == bones->neck || i == bones->neck1) {
      add_rot(orientation, 0.0, 1.0, 0.0, gait->neck_yaw, orientation);
      add_rot(orientation, 0.0, 0.0, 1.0, gait->neck_yaw * 0.45, orientation);
      add_rot(orientation, 1.0, 0.0, 0.0, gait->neck_pitch, orientation);
    } else if (i == bones->head) {
      add_rot(orientation, 0.0, 1.0, 0.0, gait->neck_yaw * 0.85, orientation);
      add_rot(orientation, 0.0, 0.0, 1.0, gait->neck_yaw * 0.35, orientation);
      add_rot(orientation, 1.0, 0.0, 0.0, gait->neck_pitch * 0.5, orientation);
    } else if (i == bones->l_arm) {
      add_rot(orientation, 0.0, 0.0, 1.0, -ARM_DOWN_ANGLE, orientation);
      add_rot(orientation, 0.0, 1.0, 0.0, gait->l_shoulder_roll, orientation);
      add_rot(orientation, 0.0, 0.0, 1.0, gait->l_shoulder_yaw, orientation);
      add_rot(orientation, 1.0, 0.0, 0.0, gait->l_shoulder_pitch, orientation);
    } else if (i == bones->l_forearm) {
      add_rot(orientation, 0.0, 0.0, 1.0, gait->l_elbow, orientation);
    } else if (i == bones->l_hand) {
      add_rot(orientation, 1.0, 0.0, 0.0, hand_pitch, orientation);
    } else if (i == bones->r_arm) {
      add_rot(orientation, 0.0, 0.0, 1.0, ARM_DOWN_ANGLE, orientation);
      add_rot(orientation, 0.0, 1.0, 0.0, gait->r_shoulder_roll, orientation);
      add_rot(orientation, 0.0, 0.0, 1.0, gait->r_shoulder_yaw, orientation);
      add_rot(orientation, 1.0, 0.0, 0.0, gait->r_shoulder_pitch, orientation);
    } else if (i == bones->r_forearm) {
      add_rot(orientation, 0.0, 0.0, 1.0, -gait->r_elbow, orientation);
    } else if (i == bones->r_hand) {
      add_rot(orientation, 1.0, 0.0, 0.0, -hand_pitch, orientation);
    } else if (i == bones->l_upleg) {
      add_rot(orientation, 1.0, 0.0, 0.0, gait->l_hip_pitch, orientation);
      add_rot(orientation, 0.0, 1.0, 0.0, gait->l_hip_roll, orientation);
      add_rot(orientation, 0.0, 0.0, 1.0, gait->l_hip_yaw, orientation);
    } else if (i == bones->r_upleg) {
      add_rot(orientation, 1.0, 0.0, 0.0, gait->r_hip_pitch, orientation);
      add_rot(orientation, 0.0, 1.0, 0.0, gait->r_hip_roll, orientation);
      add_rot(orientation, 0.0, 0.0, 1.0, gait->r_hip_yaw, orientation);
    } else if (i == bones->l_leg) {
      add_rot(orientation, 1.0, 0.0, 0.0, gait->l_knee, orientation);
    } else if (i == bones->r_leg) {
      add_rot(orientation, 1.0, 0.0, 0.0, gait->r_knee, orientation);
    } else if (i == bones->l_foot) {
      add_rot(orientation, 1.0, 0.0, 0.0, gait->l_ankle_pitch, orientation);
      add_rot(orientation, 0.0, 1.0, 0.0, gait->l_ankle_roll, orientation);
    } else if (i == bones->r_foot) {
      add_rot(orientation, 1.0, 0.0, 0.0, gait->r_ankle_pitch, orientation);
      add_rot(orientation, 0.0, 1.0, 0.0, gait->r_ankle_roll, orientation);
    } else if (i == bones->l_toe) {
      add_rot(orientation, 1.0, 0.0, 0.0, gait->l_toe_pitch, orientation);
    } else if (i == bones->r_toe) {
      add_rot(orientation, 1.0, 0.0, 0.0, gait->r_toe_pitch, orientation);
    } else {
      continue;
    }

    wb_skin_set_bone_orientation(skin, i, orientation, false);
  }
}

int main(int argc, char **argv) {
  wb_robot_init();

  char *skin_device_name = NULL;
  double gait_frequency = 0.50;
  int c;
  while ((c = getopt(argc, argv, "d:g:f:s:e:l")) != -1) {
    switch (c) {
      case 'd':
        skin_device_name = optarg;
        break;
      case 'g':
        gait_frequency = atof(optarg);
        break;
      case 'f':
      case 's':
      case 'e':
      case 'l':
        break;
      default:
        print_usage(argv[0]);
        wb_robot_cleanup();
        return 1;
    }
  }

  if (!skin_device_name) {
    fprintf(stderr, "Missing -d <skin_device_name>\n");
    print_usage(argv[0]);
    wb_robot_cleanup();
    return 1;
  }

  WbDeviceTag skin = wb_robot_get_device(skin_device_name);
  const int bone_count = wb_skin_get_bone_count(skin);
  if (bone_count == 0) {
    printf("No bones found.\n");
    wb_robot_cleanup();
    return 0;
  }

  char **names = (char **)malloc(bone_count * sizeof(char *));
  double *neutral = (double *)malloc(bone_count * 4 * sizeof(double));
  if (!names || !neutral) {
    fprintf(stderr, "Failed to allocate skin metadata.\n");
    free(names);
    free(neutral);
    wb_robot_cleanup();
    return 1;
  }

  for (int i = 0; i < bone_count; ++i) {
    const char *name = wb_skin_get_bone_name(skin, i);
    names[i] = (char *)malloc(strlen(name) + 1);
    if (!names[i]) {
      fprintf(stderr, "Failed to allocate bone name.\n");
      for (int j = 0; j < i; ++j)
        free(names[j]);
      free(names);
      free(neutral);
      wb_robot_cleanup();
      return 1;
    }
    strcpy(names[i], name);

    const double *orientation = wb_skin_get_bone_orientation(skin, i, false);
    for (int j = 0; j < 4; ++j)
      neutral[i * 4 + j] = orientation[j];
  }

  const BoneIndex bones = build_bone_index(names, bone_count);
  if (bones.root < 0)
    fprintf(stderr, "Warning: Hips bone not found, root position update disabled.\n");

  printf("Visible skin skeleton: %d joints\n", bone_count);
  for (int i = 0; i < bone_count; ++i)
    printf("  Joint %02d: %s\n", i, names[i]);
  printf("Procedural DOF note: waist/hips use lateral roll plus yaw twist; arms keep a relaxed elbow bend.\n");

  SkeletonOverlay overlay;
  const bool overlay_enabled = create_skeleton_overlay(&overlay, &bones, bone_count);
  if (overlay_enabled)
    printf("Visible skeleton overlay loaded: %d joint markers, %d bone links, %d RGB axis joints.\n",
           overlay.joint_count,
           overlay.link_count,
           overlay.axis_count);

  double root_base_position[3] = {0.0, 0.0, 0.0};
  if (bones.root >= 0) {
    const double *position = wb_skin_get_bone_position(skin, bones.root, false);
    for (int j = 0; j < 3; ++j)
      root_base_position[j] = position[j];
  }

  GaitState state;
  GaitState target;
  memset(&state, 0, sizeof(GaitState));
  memset(&target, 0, sizeof(GaitState));

  const double lpf_alpha = 0.14;
  printf("Heavy procedural humanoid gait loaded: %d bones, %.2f Hz.\n", bone_count, gait_frequency);
  printf("Legacy BVH args are accepted, but BVH animation is disabled.\n");

  while (wb_robot_step(TIME_STEP) != -1) {
    const double time = wb_robot_get_time();
    const double cycle = fmod(time * gait_frequency, 1.0);

    compute_gait_target(cycle, &target);
    lpf_state(&state, &target, lpf_alpha);
    apply_gait(skin, &bones, bone_count, neutral, &state, cycle);

    if (bones.root >= 0) {
      const double forward = WALK_SPEED * time;
      double position[3] = {
        root_base_position[0] + state.pelvis_shift_x,
        root_base_position[1] + state.pelvis_bob_y,
        root_base_position[2] + forward
      };
      wb_skin_set_bone_position(skin, bones.root, position, false);
    }

    if (overlay_enabled)
      update_skeleton_overlay(skin, &overlay);
  }

  cleanup_skeleton_overlay(&overlay);
  for (int i = 0; i < bone_count; ++i)
    free(names[i]);
  free(names);
  free(neutral);
  wb_robot_cleanup();
  return 0;
}
