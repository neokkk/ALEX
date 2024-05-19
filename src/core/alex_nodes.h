// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*
 * This file contains code for ALEX nodes. There are two types of nodes in ALEX:
 * - Model nodes (equivalent to internal/inner nodes of a B+ Tree)
 * - Data nodes, sometimes referred to as leaf nodes (equivalent to leaf nodes
 * of a B+ Tree)
 */

#pragma once

#include "alex_base.h"
#include "bitmap.h"

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#define ALEX_USE_BUFFERED_INSERT 1

// Whether we store key and payload arrays separately in data nodes
// By default, we store them separately
#define ALEX_DATA_NODE_SEP_ARRAYS 1

// Whether we use lzcnt and tzcnt when manipulating a bitmap (e.g., when finding
// the closest gap).
// If your hardware does not support lzcnt/tzcnt (e.g., your Intel CPU is
// pre-Haswell), set this to 0.
#define ALEX_USE_LZCNT 1

namespace alex {

// A parent class for both types of ALEX nodes
template <class T, class P>
class AlexNode {
public:
  // Whether this node is a leaf (data) node
  bool is_leaf_ = false;

  // Power of 2 to which the pointer to this node is duplicated in its parent
  // model node
  // For example, if duplication_factor_ is 3, then there are 8 redundant
  // pointers to this node in its parent
  uint8_t duplication_factor_ = 0;

  // Node's level in the RMI. Root node is level 0
  short level_ = 0;

  // Both model nodes and data nodes nodes use models
  LinearModel<T> model_;

  // Could be either the expected or empirical cost, depending on how this field
  // is used
  double cost_ = 0.0;

  // nk. Time to find a key in this node
  double find_cost_ = 0.0;

  AlexNode() = default;
  explicit AlexNode(short level) : level_(level) {}
  AlexNode(short level, bool is_leaf) : is_leaf_(is_leaf), level_(level) {}
  virtual ~AlexNode() = default;

  // The size in bytes of all member variables in this class
  virtual long long node_size() const = 0;
};

template <class T, class P, class Alloc = std::allocator<std::pair<T, P>>>
class AlexModelNode : public AlexNode<T, P> {
public:
  typedef AlexModelNode<T, P, Alloc> self_type;
  typedef typename Alloc::template rebind<self_type>::other alloc_type;
  typedef typename Alloc::template rebind<AlexNode<T, P> *>::other pointer_alloc_type;

  const Alloc &allocator_;

  // Number of logical children. Must be a power of 2
  int num_children_ = 0;

  // Array of pointers to children
  AlexNode<T, P> **children_ = nullptr;

  explicit AlexModelNode(const Alloc& alloc = Alloc())
    : AlexNode<T, P>(0, false), allocator_(alloc) {}

  explicit AlexModelNode(short level, const Alloc& alloc = Alloc())
    : AlexNode<T, P>(level, false), allocator_(alloc) {}

  ~AlexModelNode() {
    if (children_ == nullptr) {
      return;
    }
    pointer_allocator().deallocate(children_, num_children_);
  }

  AlexModelNode(const self_type &other)
    : AlexNode<T, P>(other),
      allocator_(other.allocator_),
      num_children_(other.num_children_) {
    children_ = new (pointer_allocator().allocate(other.num_children_)) AlexNode<T, P> *[other.num_children_];
    std::copy(other.children_, other.children_ + other.num_children_, children_);
  }

  // Given a key, traverses to the child node responsible for that key
  inline AlexNode<T, P> *get_child_node(const T &key) {
    int bucketID = this->model_.predict(key);
    bucketID = std::min<int>(std::max<int>(bucketID, 0), num_children_ - 1);
    return children_[bucketID];
  }

  // Expand by a power of 2 by creating duplicates of all existing child
  // pointers.
  // Input is the base 2 log of the expansion factor, in order to guarantee
  // expanding by a power of 2.
  // Returns the expansion factor.
  int expand(int log2_expansion_factor) {
    assert(log2_expansion_factor >= 0);

    int expansion_factor = 1 << log2_expansion_factor;
    int num_new_children = num_children_ * expansion_factor;
    auto new_children = new (pointer_allocator().allocate(num_new_children)) AlexNode<T, P> *[num_new_children];
    int cur = 0;

    while (cur < num_children_) {
      AlexNode<T, P> *cur_child = children_[cur];
      int cur_child_repeats = 1 << cur_child->duplication_factor_;

      for (int i = expansion_factor * cur; i < expansion_factor * (cur + cur_child_repeats); i++) {
        new_children[i] = cur_child;
      }
      cur_child->duplication_factor_ += log2_expansion_factor;
      cur += cur_child_repeats;
    }

    pointer_allocator().deallocate(children_, num_children_);

    children_ = new_children;
    num_children_ = num_new_children;

    this->model_.expand(expansion_factor);

    return expansion_factor;
  }

  pointer_alloc_type pointer_allocator() {
    return pointer_alloc_type(allocator_);
  }

  long long node_size() const override {
    long long size = sizeof(self_type);
    size += num_children_ * sizeof(AlexNode<T, P> *);  // pointers to children
    return size;
  }

  // Helpful for debugging
  bool validate_structure(bool verbose = false) const {
    if (num_children_ == 0) {
      if (verbose) {
        std::cout << "[Childless node] addr: " << this << ", level "
                  << this->level_ << std::endl;
      }
      return false;
    }

    if (num_children_ == 1) {
      if (verbose) {
        std::cout << "[Single child node] addr: " << this << ", level "
                  << this->level_ << std::endl;
      }
      return false;
    }

    if (std::ceil(std::log2(num_children_)) !=
        std::floor(std::log2(num_children_))) {
      if (verbose) {
        std::cout << "[Num children not a power of 2] num children: "
                  << num_children_ << std::endl;
      }
      return false;
    }

    if (this->model_.a_ == 0) {
      if (verbose) {
        std::cout << "[Model node with zero slope] addr: " << this << ", level "
                  << this->level_ << std::endl;
      }
      return false;
    }

    AlexNode<T, P> *cur_child = children_[0];
    int cur_repeats = 1;
    int i;

    for (i = 1; i < num_children_; i++) {
      if (children_[i] == cur_child) {
        cur_repeats++;
      } else {
        if (cur_repeats != (1 << cur_child->duplication_factor_)) {
          if (verbose) {
            std::cout << "[Incorrect duplication factor] num actual repeats: "
                      << cur_repeats << ", num dup_factor repeats: "
                      << (1 << cur_child->duplication_factor_)
                      << ", parent addr: " << this
                      << ", parent level: " << this->level_
                      << ", parent num children: " << num_children_
                      << ", child addr: " << children_[i - cur_repeats]
                      << ", child pointer indexes: [" << i - cur_repeats << ", "
                      << i << ")" << std::endl;
          }
          return false;
        }

        if (std::ceil(std::log2(cur_repeats)) != std::floor(std::log2(cur_repeats))) {
          if (verbose) {
            std::cout
              << "[Num duplicates not a power of 2] num actual repeats: "
              << cur_repeats << std::endl;
          }
          return false;
        }

        if (i % cur_repeats != 0) {
          if (verbose) {
            std::cout
              << "[Duplicate region incorrectly aligned] num actual repeats: "
              << cur_repeats << ", num dup_factor repeats: "
              << (1 << cur_child->duplication_factor_)
              << ", child pointer indexes: [" <<  i - cur_repeats << ", " << i
              << ")" << std::endl;
          }
          return false;
        }

        cur_child = children_[i];
        cur_repeats = 1;
      }
    }

    if (cur_repeats != (1 << cur_child->duplication_factor_)) {
      if (verbose) {
        std::cout << "[Incorrect duplication factor] num actual repeats: "
                  << cur_repeats << ", num dup_factor repeats: "
                  << (1 << cur_child->duplication_factor_)
                  << ", parent addr: " << this
                  << ", parent level: " << this->level_
                  << ", parent num children: " << num_children_
                  << ", child addr: " << children_[i - cur_repeats]
                  << ", child pointer indexes: [" << i - cur_repeats << ", "
                  << i << ")" << std::endl;
      }
      return false;
    }

    if (std::ceil(std::log2(cur_repeats)) !=
        std::floor(std::log2(cur_repeats))) {
      if (verbose) {
        std::cout << "[Num duplicates not a power of 2] num actual repeats: "
                  << cur_repeats << std::endl;
      }
      return false;
    }

    if (i % cur_repeats != 0) {
      if (verbose) {
        std::cout
            << "[Duplicate region incorrectly aligned] num actual repeats: "
            << cur_repeats << ", num dup_factor repeats: "
            << (1 << cur_child->duplication_factor_)
            << ", child pointer indexes: [" << i - cur_repeats << ", " << i
            << ")" << std::endl;
      }
      return false;
    }

    if (cur_repeats == num_children_) {
      if (verbose) {
        std::cout << "[All children are the same] num actual repeats: "
                  << cur_repeats << ", parent addr: " << this
                  << ", parent level: " << this->level_
                  << ", parent num children: " << num_children_ << std::endl;
      }

      return false;
    }

    return true;
  }
};

/*
* Functions are organized into different sections:
* - Constructors and destructors
* - General helper functions
* - Iterator
* - Cost model
* - Bulk loading and model building (e.g., bulk_load, bulk_load_from_existing)
* - Lookups (e.g., find_key)
* - Inserts and resizes (e.g, insert)
* - Deletes (e.g., erase, erase_one)
* - Stats
* - Debugging
*/
template <class T, class P, class Compare = AlexCompare,
          class Alloc = std::allocator<std::pair<T, P>>,
          bool allow_duplicates = true>
class AlexDataNode : public AlexNode<T, P> {
public:
  class AlexDataBuffer;
  typedef std::pair<T, P> V;
  typedef int8_t order_t;
  typedef AlexDataNode<T, P, Compare, Alloc, allow_duplicates> self_type;
  typedef typename Alloc::template rebind<self_type>::other alloc_type;
  typedef typename Alloc::template rebind<AlexDataBuffer *>::other buffer_alloc_type;
  typedef typename Alloc::template rebind<uint64_t>::other bitmap_alloc_type;

  const Compare &key_less_;
  const Alloc &allocator_;

  // Forward declaration
  template <typename node_type = self_type, typename payload_return_type = P,
            typename value_return_type = V>
  class Iterator;
  typedef Iterator<> iterator_type;
  typedef Iterator<const self_type, const P, const V> const_iterator_type;

  self_type *next_leaf_ = nullptr;
  self_type *prev_leaf_ = nullptr;

  int data_capacity_ = 0;  // size of buffer
  int num_keys_ = 0;  // number of filled buffer slots

  // Bitmap: each uint64_t represents 64 positions in reverse order
  // (i.e., each uint64_t is "read" from the right-most bit to the left-most
  // bit)
  uint64_t *bitmap_ = nullptr;
  int bitmap_size_ = 0;  // number of int64_t in bitmap

  AlexDataBuffer **buffer_ = nullptr;

  double expansion_threshold_ = 1;  // expand after m_num_keys is >= this number
  double contraction_threshold_ = 0;  // contract after m_num_keys is < this number
  static constexpr int kDefaultMaxDataNodeBytes_ = 1 << 24;  // by default, maximum data node size is 16MB
  int max_slots_ = kDefaultMaxDataNodeBytes_ / sizeof(V);  // cannot expand beyond this number of key/data slots

  // Counters used in cost models
  long long num_shifts_ = 0;                 // does not reset after resizing
  long long num_exp_search_iterations_ = 0;  // does not reset after resizing
  int num_lookups_ = 0;                      // does not reset after resizing
  int num_inserts_ = 0;                      // does not reset after resizing
  int num_resizes_ = 0;  // technically not required, but nice to have

  int id_not_equal = 0;

  // Variables for determining append-mostly behavior
  T max_key_ = std::numeric_limits<T>::lowest();  // max key in node, updates after inserts but not erases
  T min_key_ = std::numeric_limits<T>::max();  // min key in node, updates after
                                               // inserts but not erases
  int num_right_out_of_bounds_inserts_ = 0;  // number of inserts that are larger than the max key
  int num_left_out_of_bounds_inserts_ = 0;  // number of inserts that are smaller than the min key
  // Node is considered append-mostly if the fraction of inserts that are out of
  // bounds is above this threshold
  // Append-mostly nodes will expand in a manner that anticipates further
  // appends
  static constexpr double kAppendMostlyThreshold = 0.9;

  // Purely for benchmark debugging purposes
  double expected_avg_exp_search_iterations_ = 0;
  double expected_avg_shifts_ = 0;

  // Placed at the end of the key/data slots if there are gaps after the max key
  static constexpr T kEndSentinel_ = std::numeric_limits<T>::max();

  class AlexDataBuffer {
    typedef typename Alloc::template rebind<T>::other key_alloc_type;
    typedef typename Alloc::template rebind<P>::other payload_alloc_type;
    typedef typename Alloc::template rebind<V>::other value_alloc_type;
    typedef typename Alloc::template rebind<order_t>::other order_alloc_type;
  
  public:
    AlexDataNode *node_;

  #if ALEX_DATA_NODE_SEP_ARRAYS
    T *key_slots_ = nullptr;  /// holds keys
    P *payload_slots_ = nullptr;  /// holds payloads, must be same size as key_slots
  #else
    V *data_slots_ = nullptr;  /// holds key-payload pairs
  #endif

    order_t *order_slots_ = nullptr;  /// order of keys in key/data_slots

    uint64_t *delete_bitmap_ = nullptr;

    order_t count = 0; /// number of keys in key/data_slots
    order_t min_idx = 0; /// index of the minimum key in key/data_slots
                         /// start point for sorted lookups
    order_t max_idx = 0; /// index of the maximum key in key/data_slots
                         /// end point for sorted lookups
    uint8_t current_collision_factor = 2; /// 2, 3, 4
  
    static constexpr uint8_t kMaxCollisionFactor = 4;

    AlexDataBuffer(AlexDataNode *node) : node_(node) {
      initialize();
    }

    explicit AlexDataBuffer(AlexDataNode *node, T key, P payload) : node_(node) {
      initialize();
      key_slots_[0] = key;
      payload_slots_[0] = payload;
      order_slots_[0] = -1;
      count++;
    }

    virtual ~AlexDataBuffer() {
      auto size = 1 << current_collision_factor;

    #if ALEX_DATA_NODE_SEP_ARRAYS
      if (key_slots_ == nullptr) {
        return;
      }
      key_allocator().deallocate(key_slots_, size);
      payload_allocator().deallocate(payload_slots_, size);
    #else
      if (data_slots_ == nullptr) {
        return;
      }
      value_allocator().deallocate(data_slots_, size);
    #endif
      order_allocator().deallocate(order_slots_, size);
      auto bitmap_size = static_cast<size_t>(std::ceil(size / 64.));
      node_->bitmap_allocator().deallocate(delete_bitmap_, bitmap_size);
    }

private:
    void initialize() {
      auto size = 1 << current_collision_factor;

    #if ALEX_DATA_NODE_SEP_ARRAYS
      key_slots_ = new (key_allocator().allocate(size)) T[size];
      payload_slots_ = new (payload_allocator().allocate(size)) P[size];
    #else
      data_slots_ = new (value_allocator().allocate(size)) V[size];
    #endif
      order_slots_ = new (order_allocator().allocate(size)) order_t[size](); /// initialize to 0

      auto bitmap_size = static_cast<size_t>(std::ceil(size / 64.));
      delete_bitmap_ = new (node_->bitmap_allocator().allocate(bitmap_size)) uint64_t[bitmap_size]();
    }

    /*** Allocators ***/

    key_alloc_type key_allocator() {
      return key_alloc_type(node_->allocator_);
    }

    payload_alloc_type payload_allocator() {
      return payload_alloc_type(node_->allocator_);
    }

  #if ALEX_DATA_NODE_SEP_ARRAYS
    value_alloc_type value_allocator() {
      return value_alloc_type(node_->allocator_);
    }
  #endif

    order_alloc_type order_allocator() {
      return order_alloc_type(node_->allocator_);
    }

public:
    /*** General helper functions ***/

    inline T get_key(int8_t pos = 0) const {
      if (count == 0) return std::numeric_limits<T>::max();
      return key_slots_[pos];
    }

    inline T *get_keys() const {
      if (count == 0) return nullptr;
      return key_slots_;
    }

    inline T get_min_key() const {
      if (count == 0) return std::numeric_limits<T>::min();
      return key_slots_[min_idx];
    }

    inline T get_max_key() const {
      if (count == 0) return std::numeric_limits<T>::max();
      return key_slots_[max_idx];
    }

    inline P *get_payload(int8_t pos = 0) const {
      if (count == 0) return nullptr;
      return &(payload_slots_[pos]);
    }

    inline P *get_payloads() const {
      if (count == 0) return nullptr;
      return payload_slots_;
    }

  #if !ALEX_DATA_NODE_SEP_ARRAYS
    inline V get_data() const {
      if (count == 0) return std::make_pair(std::numeric_limits<T>::max(), nullptr);
      return data_slots_[0];
    }

    inline V *get_datas() const {
      if (count == 0) return nullptr;
      return data_slots_;
    }
  #endif

    void delete_key(int8_t pos) {
      int bitmap_pos = pos >> 6;
      if (bitmap_pos >= (1 << current_collision_factor)) {
        return;
      }
      int bit_pos = pos - (bitmap_pos << 6);
      delete_bitmap_[bitmap_pos] = delete_bitmap_[bitmap_pos] | (1ULL << bit_pos);
    }

    bool is_full() {
      return count == (1 << current_collision_factor);
    }

    bool is_totally_full() {
      return (
        current_collision_factor == kMaxCollisionFactor &&
        is_full()
      );
    }

    int num_keys_in_range(const T &left, const T &right) const {
      int num_keys = 0;
      int start_pos = find_lower_bound(left);
      start_pos = key_slots_[start_pos] == left ? start_pos : start_pos + 1; /// greater equal than left
      order_t next_order = order_slots_[start_pos];
      int next_key = next_order == -1 ? right : next_order;

      while (next_key != right) {
        num_keys++;
        next_order = order_slots_[next_key];
        next_key = next_order == -1 ? right : next_order;
      }

      return num_keys;
    }

    order_t size() {
      return count;
    }

    size_t capacity() {
      return 1 << current_collision_factor;
    }

    int find_idx(T key) {
      int size = 1 << current_collision_factor;

      printf("count: %d\n", count);
      if (count == 0 ||
        node_->key_less(key, key_slots_[min_idx]) ||
        node_->key_less(key_slots_[max_idx], key)) {
        return -1;
      }

    #ifdef __AVX512F__
      std::cout << "simd lookup" << std::endl;
      const int unit_count = 512 / (sizeof(T) * byte);
      const int vec_size = size / unit_count;
      int i = 0;
      __m512i vkey = _mm512_set1_epi64(key);

      for (; i < vec_size; i++) {
        __m512i vkeys = _mm512_loadu_epi64(&key_slots_[i * unit_count]);
        __mmask16 mask = _mm512_cmpeq_epi64_mask(vkeys, vkey);

        if (mask != 0) { /// If any keys match
          int match_key = _tzcnt_u64(mask) + i; /// Find the index of the first match
          __m512i match_vec = _mm512_set1_epi64(match_key);
          __mmask16 match_mask = _mm512_cmpeq_epi64_mask(vkeys, match_vec);
          int match_idx = __tzcnt_u64(match_mask) + i * unit_count;
          return match_idx;
        }
      }

      /// Handle remaining elements
      for (int j = i * unit_count; j < size; j++) {
        if (node_->key_equal(key, key_slots_[j])) {
          return j;
        }
      }
    #else
      for (int i = 0; i < count; i++) {
        if (node_->key_equal(key, key_slots_[i])) {
          return i;
        }
      }
    #endif
      
      return -1;
    }

    bool key_exists(T key) {
      return find_idx(key) != -1;
    }

    P *lookup(T key) {
      int idx = find_idx(key);
      if (idx < 0) {
        return nullptr;
      }
      return &payload_slots_[idx];
    }

    bool append(const T &key, const P &payload) {
      /// TODO: transaction start
      if (is_full()) {
        std::cout << "is full" << std::endl;
        bool result = expand();
        /// TODO: transaction end
        if (!result) { /// full
          return false;
        }
      }

      auto pos = size();
      printf("pos: %d\n", pos);
      key_slots_[pos] = key;
      payload_slots_[pos] = payload;

      /*** Update order ***/

      printf("input key: %lld, min key: %lld, max key: %lld\n", key, key_slots_[min_idx], key_slots_[max_idx]);
      printf("min_idx: %d, max_idx: %d\n", min_idx, max_idx);

      if (node_->key_less(key, key_slots_[min_idx])) {
        std::cout << "key is less than min" << std::endl;
        order_slots_[pos] = min_idx;
        min_idx = pos;
      } else if (node_->key_less(key_slots_[max_idx], key)) {
        std::cout << "key is greater than max" << std::endl;
        order_slots_[max_idx] = pos;
        max_idx = pos;
        order_slots_[pos] = -1; /// update max to -1
      } else { /// TODO: concurrent control
        std::cout << "key is in the middle" << std::endl;
        auto order_idx = find_last_no_greater_than(key, true);
        order_slots_[pos] = order_slots_[order_idx];
        order_slots_[order_idx] = pos;
      }

      for (int i = 0; i <= pos; i++) {
        printf("%d ", i, order_slots_[i]);
      }
      printf("\n");

      count++;

      return true;
    }

    order_t find_last_no_greater_than(T key, bool use_simd = false) {
      int size = 1 << current_collision_factor;

      if (use_simd) {
        assert(size > 4);

        const int unit_count = 512 / (sizeof(T) * byte);
        const int vec_size = size / unit_count;
        T max_key = std::numeric_limits<T>::min();
        order_t max_idx = -1;
        __m512i vkey = _mm512_set1_epi64(key);

        for (int i = 0; i < vec_size; i++) {
          __m512i vkeys = _mm512_load_epi64(&key_slots_[i * unit_count]);
          __mmask16 mask = _mm512_cmp_epi64_mask(vkeys, vkey, _MM_CMPINT_LT); /// Compare keys for less than

          if (mask != 0) { /// If any keys are less than the given key
            int match_key = _mm512_mask_reduce_max_epi64(mask, vkeys);
            if (match_key > max_key) {
              max_key = match_key;
              __m512i match_vec = _mm512_set1_epi64(match_key);
              __mmask16 match_mask = _mm512_cmpeq_epi64_mask(vkeys, match_vec);
              max_idx = __tzcnt_u64(match_mask) + i * unit_count;
            }
          }
        }

        for (int i = vec_size * unit_count; i < size; i++) {
          if (node_->key_less(key_slots_[i], key)) {
            max_key = key_slots_[i];
            max_idx = i;
          }
        }

        return max_idx;
      } else {
        std::vector<order_t> small_key_idxs;
        printf("small_key_idxs(count: %d): ", count);
        for (int i = 0; i < count; i++) {
          if (node_->key_less(key, key_slots_[i])) {
            printf("%d ", key_slots_[i]);
            small_key_idxs.push_back(i);
          }
        }
        printf("\n");
        auto result = std::max_element(small_key_idxs.begin(), small_key_idxs.end(), [&](order_t a, order_t b) {
          return node_->key_less(key_slots_[a], key_slots_[b]);
        });
        if (result == small_key_idxs.end()) {
          return -1;
        }
        printf("result: %d\n");
        return *result;
      }
    }

    order_t find_first_greater_than(T key, bool use_simd = false) {
      int size = 1 << current_collision_factor;

      if (use_simd) {
        assert(size > 4);

        const int unit_count = 512 / (sizeof(T) * byte);
        const int vec_size = size / unit_count;
        T min_key = std::numeric_limits<T>::max();
        order_t min_idx = -1;
        __m512i vkey = _mm512_set1_epi64(key);

        for (int i = 0; i < vec_size; i++) {
          __m512i vkeys = _mm512_load_epi64(&key_slots_[i * unit_count]);
          __mmask16 mask = _mm512_cmp_epi64_mask(vkeys, vkey, _MM_CMPINT_GT); /// Compare keys for greater than

          if (mask != 0) { /// If any keys are greater than the given key
            int match_key = _mm512_mask_reduce_min_epi64(mask, vkeys);
            if (match_key < min_key) {
              min_key = match_key;
              __m512i match_vec = _mm512_set1_epi64(match_key);
              __mmask16 match_mask = _mm512_cmpeq_epi64_mask(vkeys, match_vec);
              min_idx = __tzcnt_u64(match_mask) + i * unit_count;
            }
          }
        }

        return min_idx;
      } else {
        std::vector<order_t> large_key_idxs;
        for (int i = 0; i < count; i++) {
          if (node_->key_less(key_slots_[i], key)) {
            large_key_idxs.push_back(i);
          }
        }
        auto result = std::min_element(large_key_idxs.begin(), large_key_idxs.end(), [&](order_t a, order_t b) {
          return node_->key_less(key_slots_[a], key_slots_[b]);
        });
        if (result == large_key_idxs.end()) {
          return -1;
        }
        return *result;
      }
    }

    /**
     * when expand, deleted keys will be removed
    */
    bool expand() {
      if (is_totally_full()) {
        std::cout << "is totally full" << std::endl;
        return false;
      }

      int prev_size = 1 << current_collision_factor;
      current_collision_factor++;
      int size = 1 << current_collision_factor;
      std::cout << "buffer expanded to " << size << std::endl;

      if (size > 8) {
        compress(); /// compress before expand
      }
      expand_without_compress(size, prev_size);

      return true;
    }

    bool expand_without_compress(int size, int prev_size) {
    #if ALEX_DATA_NODE_SEP_ARRAYS
      auto new_key_slots = new (key_allocator().allocate(size)) T[size];
      memcpy(new_key_slots, key_slots_, prev_size * sizeof(T));
      key_allocator().deallocate(key_slots_, prev_size);
      key_slots_ = new_key_slots;

      auto new_payload_slots = new (payload_allocator().allocate(size)) P[size];
      memcpy(new_payload_slots, payload_slots_, prev_size * sizeof(P));
      payload_allocator().deallocate(payload_slots_, prev_size);
      payload_slots_ = new_payload_slots;
    #else
      auto new_data_slots = new (value_allocator().allocate(size)) V[size];
      memcpy(new_data_slots, data_slots_, prev_size * sizeof(V));
      value_allocator().deallocate(data_slots_, prev_size);
      data_slots_ = new_data_slots;
    #endif

      auto new_order_slots = new (order_allocator().allocate(size)) order_t[size]();
      memcpy(new_order_slots, order_slots_, prev_size * sizeof(order_t));
      order_allocator().deallocate(order_slots_, prev_size);
      order_slots_ = new_order_slots;

      auto bitmap_size = static_cast<size_t>(std::ceil(size / 64.));
      auto new_delete_bitmap = new (node_->bitmap_allocator().allocate(bitmap_size)) uint64_t[bitmap_size]();
      memcpy(new_delete_bitmap, delete_bitmap_, bitmap_size);
      node_->bitmap_allocator().deallocate(delete_bitmap_, bitmap_size);
      delete_bitmap_ = new_delete_bitmap;

      return true;
    }

    int compress() {
      auto size = 1 << current_collision_factor;
      uint16_t delete_bits = static_cast<uint16_t>(*delete_bitmap_ >> (64 - size));
      int popcount = __builtin_popcount(delete_bits);
      uint16_t mask = ~delete_bits << (16 - size);
      T min_key = std::numeric_limits<T>::max(), max_key = std::numeric_limits<T>::min();
      int min_key_idx = 0;

      assert(size > 8);

      if (popcount == 0) {
        return 0;
      }

      if (popcount == size) {
        auto bitmap_size = static_cast<size_t>(std::ceil(size / 64.));
        memset(delete_bitmap_, 0, bitmap_size);
        memset(order_slots_, 0, size * sizeof(order_t));
        count = 0;
        min_idx = 0;
        max_idx = 0;
        return size;
      }

    #if ALEX_DATA_NODE_SEP_ARRAYS
      int key_vec_size = sizeof(T) * size / 64;
      int payload_vec_size = sizeof(P) * size / 64;
      std::cout << "key_vec_size: " << key_vec_size << ", payload_vec_size: " << payload_vec_size << std::endl;
      int last_key_attr = 0, last_payload_addr = 0;

      for (int i = 0; i < key_vec_size; ++i) {
        int right_shift = size / key_vec_size * (key_vec_size - i - 1);
        __mmask16 key_mask = reverse_bits(mask >> right_shift) & (1 << (size / key_vec_size) - 1);
        size_t valid_key_count = __builtin_popcount(key_mask);

        if (valid_key_count == 0) {
          continue;
        }

        __m512i key_vec = _mm512_load_epi64(key_slots_ + i * byte * sizeof(T));
        __m512i key_res_vec = _mm512_maskz_compress_epi64(key_mask, key_vec);

        int min_key_ = reduce_min_exclude_zero_SIMD(key_res_vec);
        if (min_key_ < min_key) {
          min_key = min_key_;
          __m512i min_vec = _mm512_set1_epi64(min_key_);
          __mmask16 min_mask = _mm512_cmpeq_epi64_mask(key_res_vec, min_vec);
          min_key_idx = __tzcnt_u64(min_mask);
        }

        auto key_res = (T *)&key_res_vec;

        memcpy(key_slots_ + last_key_attr, key_res, valid_key_count * sizeof(T));
        last_key_attr += valid_key_count * sizeof(T);
      }
      std::cout << "min_key: " << min_key << ", max_key: " << max_key << std::endl;
      
      for (int i = 0; i < payload_vec_size; ++i) {
        int right_shift = size / payload_vec_size * (payload_vec_size - i - 1);
        __mmask16 payload_mask = reverse_bits(mask >> right_shift) & (1 << (size / payload_vec_size) - 1);
        size_t valid_payload_count = __builtin_popcount(payload_mask);

        if (valid_payload_count == 0) {
          continue;
        }

        __m512i payload_vec = _mm512_load_epi64(payload_slots_ + i * byte * sizeof(P));
        __m512i payload_res_vec = _mm512_maskz_compress_epi64(payload_mask, payload_vec);
        auto payload_res = (P *)&payload_res_vec;

        memcpy(payload_slots_ + last_payload_addr, payload_res, valid_payload_count * sizeof(P));
        last_payload_addr += valid_payload_count * sizeof(P);
      }

    #else
      int value_vec_size = sizeof(V) * size / 64;
      int last_data_addr = 0;

      for (int i = 0; i < value_vec_size; ++i) {
        int right_shift = size / value_vec_size * (value_vec_size - i - 1);
        __mmask16 value_mask = reverse_bits(mask >> right_shift) & (1 << (size / value_vec_size) - 1);
        size_t valid_value_count = __builtin_popcount(value_mask);

        if (valid_value_count == 0) {
          continue;
        }

        __mm512i value_vec = _mm512_load_epi64(data_slots_ + i * byte * sizeof(V));
        __mm512i value_res_vec = _mm512_maskz_compress_epi64(value_mask, value_vec);
        auto value_res = (V *)&value_res_vec;

        memcpy(data_slots_ + last_data_addr, value_res, valid_value_count * sizeof(V));
        last_data_addr += valid_value_count * sizeof(V);
      }
    #endif

      count = size - popcount;

      /* Update order */

      memset(order_slots_, 0, size * sizeof(order_t));
      min_idx = min_key_idx;

      order_t match_idx;

      for (int i = 0; i < size - 1; i++) {
        match_idx = find_first_greater_than(key_slots_[min_key_idx], true);
        order_slots_[min_key_idx] = match_idx;
        min_key_idx = match_idx;
      }

      match_idx = find_first_greater_than(key_slots_[min_key_idx], true);
      order_slots_[min_key_idx] = match_idx;
      order_slots_[match_idx] = -1;
      max_idx = match_idx;

      /*** Initial bitmap ***/

      auto bitmap_size = static_cast<size_t>(std::ceil(size / 64.));
      memset(delete_bitmap_, 0, bitmap_size);

      return popcount;
    }

    double utilization() {
      return count / (1 << current_collision_factor);
    }

    void print() {
      for (int i = 0; i < count; i++) {
        std::cout << key_slots_[i] << " ";
      }
    }
  };

  /*** Constructors and destructors ***/

  explicit AlexDataNode(const Compare &comp = Compare(), const Alloc &alloc = Alloc())
    : AlexNode<T, P>(0, true), key_less_(comp), allocator_(alloc) {}

  AlexDataNode(short level, int max_data_node_slots,
               const Compare &comp = Compare(), const Alloc &alloc = Alloc())
    : AlexNode<T, P>(level, true),
      key_less_(comp),
      allocator_(alloc),
      max_slots_(max_data_node_slots) {}

  ~AlexDataNode() {
    if (buffer_ == nullptr) {
      return;
    }
    buffer_allocator().deallocate(buffer_, data_capacity_);
    bitmap_allocator().deallocate(bitmap_, bitmap_size_);
  }

  AlexDataNode(const self_type &other)
    : AlexNode<T, P>(other),
      key_less_(other.key_less_),
      allocator_(other.allocator_),
      next_leaf_(other.next_leaf_),
      prev_leaf_(other.prev_leaf_),
      data_capacity_(other.data_capacity_),
      num_keys_(other.num_keys_),
      bitmap_size_(other.bitmap_size_),
      expansion_threshold_(other.expansion_threshold_),
      contraction_threshold_(other.contraction_threshold_),
      max_slots_(other.max_slots_),
      num_shifts_(other.num_shifts_),
      num_exp_search_iterations_(other.num_exp_search_iterations_),
      num_lookups_(other.num_lookups_),
      num_inserts_(other.num_inserts_),
      num_resizes_(other.num_resizes_),
      max_key_(other.max_key_),
      min_key_(other.min_key_),
      num_right_out_of_bounds_inserts_(other.num_right_out_of_bounds_inserts_),
      num_left_out_of_bounds_inserts_(other.num_left_out_of_bounds_inserts_),
      expected_avg_exp_search_iterations_(other.expected_avg_exp_search_iterations_),
      expected_avg_shifts_(other.expected_avg_shifts_) {
  }

    /*** Allocators ***/

    bitmap_alloc_type bitmap_allocator() {
      return bitmap_alloc_type(allocator_);
    }

    buffer_alloc_type buffer_allocator() {
      return buffer_alloc_type(allocator_);
    }

  // Check whether the position corresponds to a key (as opposed to a gap)
  inline bool check_exists(int pos) const {
    assert(pos >= 0 && pos < data_capacity_);
    int bitmap_pos = pos >> 6;
    int bit_pos = pos - (bitmap_pos << 6);
    return static_cast<bool>(bitmap_[bitmap_pos] & (1ULL << bit_pos));
  }

  // Mark the entry for position in the bitmap
  inline void set_bit(int pos) {
    assert(pos >= 0 && pos < data_capacity_);
    int bitmap_pos = pos >> 6;
    int bit_pos = pos - (bitmap_pos << 6);
    bitmap_[bitmap_pos] |= (1ULL << bit_pos);
  }

  // Mark the entry for position in the bitmap
  inline void set_bit(uint64_t bitmap[], int pos) {
    int bitmap_pos = pos >> 6;
    int bit_pos = pos - (bitmap_pos << 6);
    bitmap[bitmap_pos] |= (1ULL << bit_pos);
  }

  // Unmark the entry for position in the bitmap
  inline void unset_bit(int pos) {
    assert(pos >= 0 && pos < data_capacity_);
    int bitmap_pos = pos >> 6;
    int bit_pos = pos - (bitmap_pos << 6);
    bitmap_[bitmap_pos] &= ~(1ULL << bit_pos);
  }

  // Value of first (i.e., min) key
  T first_key() const {
    for (int i = 0; i < data_capacity_; i++) {
      if (check_exists(i))
        return buffer_[i]->get_min_key();
    }
    return std::numeric_limits<T>::max();
  }

  // Value of last (i.e., max) key
  T last_key() const {
    for (int i = data_capacity_ - 1; i >= 0; i--) {
      if (check_exists(i))
        return buffer_[i]->get_max_key();
    }
    return std::numeric_limits<T>::lowest();
  }

  // Position in key/data_slots of first (i.e., min) key
  int first_pos() const {
    for (int i = 0; i < data_capacity_; i++) {
      if (check_exists(i)) return i;
    }
    return 0;
  }

  // Position in key/data_slots of last (i.e., max) key
  int last_pos() const {
    for (int i = data_capacity_ - 1; i >= 0; i--) {
      if (check_exists(i)) return i;
    }
    return 0;
  }

  // Number of keys between positions left and right (exclusive) in
  // key/data_slots
  int num_keys_in_range(int left, int right) const {
    assert(left >= 0 && left <= right && right <= data_capacity_);
    int num_keys = 0;
    int left_bitmap_idx = left >> 6;
    int right_bitmap_idx = right >> 6;
    
    for (int i = left_bitmap_idx; i <= right_bitmap_idx; i = get_next_filled_buffer(i, true)) {
      num_keys += buffer_[i]->size();
    }
    
    return num_keys;
  }

  bool key_exists(const T &key) const {
    int position = predict_position(key);
    if (position < data_capacity_) {
      return buffer_[position]->key_exists(key);
    }
    return false;
  }

  // True if a < b
  template <class K>
  forceinline bool key_less(const T &a, const K &b) const {
    return key_less_(a, b);
  }

  // True if a <= b
  template <class K>
  forceinline bool key_lessequal(const T &a, const K &b) const {
    return !key_less_(b, a);
  }

  // True if a > b
  template <class K>
  forceinline bool key_greater(const T &a, const K &b) const {
    return key_less_(b, a);
  }

  // True if a >= b
  template <class K>
  forceinline bool key_greaterequal(const T &a, const K &b) const {
    return !key_less_(a, b);
  }

  // True if a == b
  template <class K>
  forceinline bool key_equal(const T &a, const K &b) const {
    return !key_less_(a, b) && !key_less_(b, a);
  }

  double mean_utilization() {
    int i = 0;
    double total_utilization = 0;

    const_iterator_type it(this, 0);
    for (; !it.is_end(); it++, i++) {
      total_utilization += it->node_->utilization();
    }
    return i == 0 ? 0 : total_utilization / i;
  }

  double variance_utilization() {
    int i = 0;
    double total_variance = 0;
    double mean = mean_utilization();

    const_iterator_type it(this, 0);
    for (; !it.is_end(); it++, i++) {
      auto utilization = it->node_->utilization();
      total_variance += (utilization - mean) * (utilization - mean);
    }
    return i == 0 ? 0 : total_variance / i;
  }

  /*** Iterator ***/

  // Forward iterator meant for iterating over a single data node.
  // By default, it is a "normal" non-const iterator.
  // Can be templated to be a const iterator.
  template <typename node_type, typename payload_return_type,
            typename value_return_type>
  class Iterator {
  public:
    node_type *node_;
    int cur_idx_ = 0;  // current position in buffer, -1 if at end
    int cur_bitmap_idx_ = 0;  // current position in bitmap
    uint64_t cur_bitmap_data_ = 0;  // caches the relevant data in the current bitmap position

    explicit Iterator(node_type *node) : node_(node) {}

    Iterator(node_type *node, int idx) : node_(node), cur_idx_(idx) {
      initialize();
    }

    void initialize() {
      cur_bitmap_idx_ = cur_idx_ >> 6;
      cur_bitmap_data_ = node_->bitmap_[cur_bitmap_idx_];

      // Zero out extra bits
      int bit_pos = cur_idx_ - (cur_bitmap_idx_ << 6);
      cur_bitmap_data_ &= ~((1ULL << bit_pos) - 1);

      (*this)++;
    }

    void operator++(int) {
      while (cur_bitmap_data_ == 0) {
        cur_bitmap_idx_++;
        if (cur_bitmap_idx_ >= node_->bitmap_size_) {
          cur_idx_ = -1;
          return;
        }
        cur_bitmap_data_ = node_->bitmap_[cur_bitmap_idx_];
      }
      
      uint64_t bit = extract_rightmost_one(cur_bitmap_data_);
      cur_idx_ = get_offset(cur_bitmap_idx_, bit);
      cur_bitmap_data_ = remove_rightmost_one(cur_bitmap_data_);
    }

    AlexDataBuffer *operator*() const {
      return node_->buffer_[cur_idx_];
    }

    AlexDataBuffer *buffer() const {
      return node_->buffer_[cur_idx_];
    }

    const T key() const {
      return buffer()->get_key();
    }

    const T *keys() const {
      if (node_->bitmap_[cur_bitmap_idx_] == 0) {
        return nullptr;
      }

      return const_cast<T *>(buffer()->get_keys());
    }

    payload_return_type payload() const {
      #if ALEX_DATA_NODE_SEP_ARRAYS
        return buffer()->get_payload();
      #else
        return buffer()->get_data().second;
      #endif
    }

    payload_return_type *payloads() const {
    #if ALEX_DATA_NODE_SEP_ARRAYS
      return buffer()->get_payloads();
    #else
      return buffer()->get_datas();
    #endif
    }

    bool is_end() const { return cur_idx_ == -1; }

    bool operator==(const Iterator &rhs) const { return cur_idx_ == rhs.cur_idx_; }

    bool operator!=(const Iterator &rhs) const { return !(*this == rhs); };
  };

  iterator_type begin() { return iterator_type(this, 0); }

  /*** Cost model ***/

  // Empirical average number of shifts per insert
  double shifts_per_insert() const {
    if (num_inserts_ == 0) {
      return 0;
    }
    return num_shifts_ / static_cast<double>(num_inserts_);
  }

  // Empirical average number of exponential search iterations per operation
  // (either lookup or insert)
  double exp_search_iterations_per_operation() const {
    if (num_inserts_ + num_lookups_ == 0) {
      return 0;
    }
    return num_exp_search_iterations_ / static_cast<double>(num_inserts_ + num_lookups_);
  }

  double empirical_cost() const {
    if (num_inserts_ + num_lookups_ == 0) {
      return 0;
    }
    double frac_inserts = static_cast<double>(num_inserts_) / (num_inserts_ + num_lookups_);
    return (
      kExpSearchIterationsWeight * exp_search_iterations_per_operation() +
      kShiftsWeight * shifts_per_insert() * frac_inserts);
  }

  // Empirical fraction of operations (either lookup or insert) that are inserts
  double frac_inserts() const {
    int num_ops = num_inserts_ + num_lookups_;
    if (num_ops == 0) {
      return 0;  // if no operations, assume no inserts
    }
    return static_cast<double>(num_inserts_) / (num_inserts_ + num_lookups_);
  }

  void reset_stats() {
    num_shifts_ = 0;
    num_exp_search_iterations_ = 0;
    num_lookups_ = 0;
    num_inserts_ = 0;
    num_resizes_ = 0;
  }

  // Computes the expected cost of the current node
  double compute_expected_cost(double frac_inserts = 0) {
    if (num_keys_ == 0) {
      return 0;
    }

    ExpectedSearchIterationsAccumulator search_iters_accumulator;
    ExpectedShiftsAccumulator shifts_accumulator(data_capacity_);
    const_iterator_type it(this, 0);
    
    for (; !it.is_end(); it++) {
      int predicted_position = std::max(0, std::min(data_capacity_ - 1, this->model_.predict(it.key())));
      search_iters_accumulator.accumulate(it.cur_idx_, predicted_position);
      shifts_accumulator.accumulate(it.cur_idx_, predicted_position);
    }
    expected_avg_exp_search_iterations_ = search_iters_accumulator.get_stat();
    expected_avg_shifts_ = shifts_accumulator.get_stat();
    double cost =
      kExpSearchIterationsWeight * expected_avg_exp_search_iterations_ +
      kShiftsWeight * expected_avg_shifts_ * frac_inserts;
    return cost;
  }

  // Computes the expected cost of a data node constructed using the input dense
  // array of keys
  // Assumes existing_model is trained on the dense array of keys
  static double compute_expected_cost(
      const V *values, int num_keys, double density,
      double expected_insert_frac,
      const LinearModel<T> *existing_model = nullptr, bool use_sampling = false,
      DataNodeStats *stats = nullptr) {
    if (use_sampling) {
      return compute_expected_cost_sampling(values, num_keys, density, expected_insert_frac, existing_model, stats);
    }

    if (num_keys == 0) {
      return 0;
    }

    int data_capacity = std::max(static_cast<int>(num_keys / density), num_keys + 1);

    // Compute what the node's model would be
    LinearModel<T> model;
    if (existing_model == nullptr) {
      build_model(values, num_keys, &model);
    } else {
      model.a_ = existing_model->a_;
      model.b_ = existing_model->b_;
    }
    model.expand(static_cast<double>(data_capacity) / num_keys);

    // Compute expected stats in order to compute the expected cost
    double cost = 0;
    double expected_avg_exp_search_iterations = 0;
    double expected_avg_shifts = 0;

    if (expected_insert_frac == 0) {
      ExpectedSearchIterationsAccumulator acc;
      build_node_implicit(values, num_keys, data_capacity, &acc, &model);
      expected_avg_exp_search_iterations = acc.get_stat();
    } else {
      ExpectedIterationsAndShiftsAccumulator acc(data_capacity);
      build_node_implicit(values, num_keys, data_capacity, &acc, &model);
      expected_avg_exp_search_iterations = acc.get_expected_num_search_iterations();
      expected_avg_shifts = acc.get_expected_num_shifts();
    }

    cost = kExpSearchIterationsWeight * expected_avg_exp_search_iterations +
      kShiftsWeight * expected_avg_shifts * expected_insert_frac;

    if (stats) {
      stats->num_search_iterations = expected_avg_exp_search_iterations;
      stats->num_shifts = expected_avg_shifts;
    }

    return cost;
  }

  // Helper function for compute_expected_cost
  // Implicitly build the data node in order to collect the stats
  static void build_node_implicit(const V *values, int num_keys,
                                  int data_capacity, StatAccumulator *acc,
                                  const LinearModel<T> *model) {
    int last_position = -1;
    int keys_remaining = num_keys;

    for (int i = 0; i < num_keys; i++) {
      int predicted_position = std::max(0, std::min(data_capacity - 1, model->predict(values[i].first)));
      int actual_position = std::max<int>(predicted_position, last_position + 1);
      int positions_remaining = data_capacity - actual_position;

      if (positions_remaining < keys_remaining) {
        actual_position = data_capacity - keys_remaining;
        for (int j = i; j < num_keys; j++) {
          predicted_position = std::max(0, std::min(data_capacity - 1, model->predict(values[j].first)));
          acc->accumulate(actual_position, predicted_position);
          actual_position++;
        }
        break;
      }

      acc->accumulate(actual_position, predicted_position);
      last_position = actual_position;
      keys_remaining--;
    }
  }

  // Using sampling, approximates the expected cost of a data node constructed
  // using the input dense array of keys
  // Assumes existing_model is trained on the dense array of keys
  // Uses progressive sampling: keep increasing the sample size until the
  // computed stats stop changing drastically
  static double compute_expected_cost_sampling(
    const V *values, int num_keys, double density,
    double expected_insert_frac,
    const LinearModel<T> *existing_model = nullptr,
    DataNodeStats *stats = nullptr) {
    const static int min_sample_size = 25;

    // Stop increasing sample size if relative diff of stats between samples is
    // less than this
    const static double rel_diff_threshold = 0.2;

    // Equivalent threshold in log2-space
    const static double abs_log2_diff_threshold = std::log2(1 + rel_diff_threshold);

    // Increase sample size by this many times each iteration
    const static int sample_size_multiplier = 2;

    // If num_keys is below this threshold, we compute entropy exactly
    const static int exact_computation_size_threshold =
        (min_sample_size * sample_size_multiplier * sample_size_multiplier * 2);

    // Target fraction of the keys to use in the initial sample
    const static double init_sample_frac = 0.01;

    // If the number of keys is sufficiently small, we do not sample
    if (num_keys < exact_computation_size_threshold) {
      return compute_expected_cost(values, num_keys, density, expected_insert_frac, existing_model, false, stats);
    }

    LinearModel<T> model;  // trained for full dense array
    if (existing_model == nullptr) {
      build_model(values, num_keys, &model);
    } else {
      model.a_ = existing_model->a_;
      model.b_ = existing_model->b_;
    }

    // Compute initial sample size and step size
    // Right now, sample_num_keys holds the target sample num keys
    int sample_num_keys = std::max(static_cast<int>(num_keys * init_sample_frac), min_sample_size);
    int step_size = 1;
    double tmp_sample_size = num_keys;  // this helps us determine the right sample size

    while (tmp_sample_size >= sample_num_keys) {
      tmp_sample_size /= sample_size_multiplier;
      step_size *= sample_size_multiplier;
    }
    step_size /= sample_size_multiplier;
    sample_num_keys = num_keys / step_size;  // now sample_num_keys is the actual sample num keys

    std::vector<SampleDataNodeStats> sample_stats;  // stats computed usinig each sample
    bool compute_shifts = expected_insert_frac != 0;  // whether we need to compute expected shifts
    double log2_num_keys = std::log2(num_keys);
    double expected_full_search_iters = 0;  // extrapolated estimate for search iters on the full array
    double expected_full_shifts = 0;  // extrapolated estimate shifts on the full array
    bool search_iters_computed = false;  // set to true when search iters is accurately computed
    bool shifts_computed = false;  // set to true when shifts is accurately computed

    // Progressively increase sample size
    while (true) {
      int sample_data_capacity = std::max(static_cast<int>(sample_num_keys / density), sample_num_keys + 1);
      LinearModel<T> sample_model(model.a_, model.b_);
      sample_model.expand(static_cast<double>(sample_data_capacity) / num_keys);

      // Compute stats using the sample
      if (expected_insert_frac == 0) {
        ExpectedSearchIterationsAccumulator acc;
        build_node_implicit_sampling(values, num_keys, sample_num_keys, sample_data_capacity, step_size, &acc, &sample_model);
        sample_stats.push_back({std::log2(sample_num_keys), acc.get_stat(), 0});
      } else {
        ExpectedIterationsAndShiftsAccumulator acc(sample_data_capacity);
        build_node_implicit_sampling(values, num_keys, sample_num_keys, sample_data_capacity, step_size, &acc, &sample_model);
        sample_stats.push_back({std::log2(sample_num_keys),
                                acc.get_expected_num_search_iterations(),
                                std::log2(acc.get_expected_num_shifts())});
      }

      if (sample_stats.size() >= 3) {
        // Check if the diff in stats is sufficiently small
        SampleDataNodeStats &s0 = sample_stats[sample_stats.size() - 3];
        SampleDataNodeStats &s1 = sample_stats[sample_stats.size() - 2];
        SampleDataNodeStats &s2 = sample_stats[sample_stats.size() - 1];

        // (y1 - y0) / (x1 - x0) = (y2 - y1) / (x2 - x1) --> y2 = (y1 - y0) /
        // (x1 - x0) * (x2 - x1) + y1
        double expected_s2_search_iters =
          (s1.num_search_iterations - s0.num_search_iterations) /
            (s1.log2_sample_size - s0.log2_sample_size) *
            (s2.log2_sample_size - s1.log2_sample_size) +
          s1.num_search_iterations;
        double rel_diff =
          std::abs((s2.num_search_iterations - expected_s2_search_iters) /
                    s2.num_search_iterations);

        if (rel_diff <= rel_diff_threshold || num_keys <= 2 * sample_num_keys) {
          search_iters_computed = true;
          expected_full_search_iters =
            (s2.num_search_iterations - s1.num_search_iterations) /
              (s2.log2_sample_size - s1.log2_sample_size) *
              (log2_num_keys - s2.log2_sample_size) +
            s2.num_search_iterations;
        }
        if (compute_shifts) {
          double expected_s2_log2_shifts =
            (s1.log2_num_shifts - s0.log2_num_shifts) /
              (s1.log2_sample_size - s0.log2_sample_size) *
              (s2.log2_sample_size - s1.log2_sample_size) +
            s1.log2_num_shifts;
          double abs_diff = std::abs((s2.log2_num_shifts - expected_s2_log2_shifts) / s2.log2_num_shifts);

          if (abs_diff <= abs_log2_diff_threshold ||
            num_keys <= 2 * sample_num_keys) {
            shifts_computed = true;
            double expected_full_log2_shifts =
              (s2.log2_num_shifts - s1.log2_num_shifts) /
                (s2.log2_sample_size - s1.log2_sample_size) *
                (log2_num_keys - s2.log2_sample_size) +
              s2.log2_num_shifts;
            expected_full_shifts = std::pow(2, expected_full_log2_shifts);
          }
        }

        // If diff in stats is sufficiently small, return the approximate
        // expected cost
        if ((expected_insert_frac == 0 && search_iters_computed) ||
            (expected_insert_frac > 0 && search_iters_computed &&
             shifts_computed)) {
          double cost =
            kExpSearchIterationsWeight * expected_full_search_iters +
            kShiftsWeight * expected_full_shifts * expected_insert_frac;
          if (stats) {
            stats->num_search_iterations = expected_full_search_iters;
            stats->num_shifts = expected_full_shifts;
          }
          return cost;
        }
      }

      step_size /= sample_size_multiplier;
      sample_num_keys = num_keys / step_size;
    }
  }

  // Helper function for compute_expected_cost_sampling
  // Implicitly build the data node in order to collect the stats
  // keys is the full un-sampled array of keys
  // sample_num_keys and sample_data_capacity refer to a data node that is
  // created only over the sample
  // sample_model is trained for the sampled data node
  static void build_node_implicit_sampling(const V *values, int num_keys,
                                           int sample_num_keys,
                                           int sample_data_capacity,
                                           int step_size, StatAccumulator *ent,
                                           const LinearModel<T> *sample_model) {
    int last_position = -1;
    int sample_keys_remaining = sample_num_keys;

    for (int i = 0; i < num_keys; i += step_size) {
      int predicted_position = std::max(0, std::min(sample_data_capacity - 1, sample_model->predict(values[i].first)));
      int actual_position = std::max<int>(predicted_position, last_position + 1);
      int positions_remaining = sample_data_capacity - actual_position;
      
      if (positions_remaining < sample_keys_remaining) {
        actual_position = sample_data_capacity - sample_keys_remaining;
        for (int j = i; j < num_keys; j += step_size) {
          predicted_position = std::max(0, std::min(sample_data_capacity - 1, sample_model->predict(values[j].first)));
          ent->accumulate(actual_position, predicted_position);
          actual_position++;
        }
        break;
      }
      
      ent->accumulate(actual_position, predicted_position);
      last_position = actual_position;
      sample_keys_remaining--;
    }
  }

  // Computes the expected cost of a data node constructed using the keys
  // between left and right in the
  // key/data_slots of an existing node
  // Assumes existing_model is trained on the dense array of keys
  static double compute_expected_cost_from_existing(
    const self_type *node, int left, int right, double density,
    double expected_insert_frac,
    const LinearModel<T> *existing_model = nullptr,
    DataNodeStats *stats = nullptr) {
    assert(left >= 0 && right <= node->data_capacity_);

    LinearModel<T> model;
    int num_actual_keys = 0;

    if (existing_model == nullptr) {
      const_iterator_type it(node, left);
      LinearModelBuilder<T> builder(&model);

      for (int i = 0; it.cur_idx_ < right && !it.is_end(); it++, i++) {
        builder.add(it.key(), i);
        num_actual_keys++;
      }
      builder.build();
    } else {
      num_actual_keys = node->num_keys_in_range(left, right);
      model.a_ = existing_model->a_;
      model.b_ = existing_model->b_;
    }

    if (num_actual_keys == 0) {
      return 0;
    }

    int data_capacity = std::max(static_cast<int>(num_actual_keys / density), num_actual_keys + 1);
    model.expand(static_cast<double>(data_capacity) / num_actual_keys);

    // Compute expected stats in order to compute the expected cost
    double cost = 0;
    double expected_avg_exp_search_iterations = 0;
    double expected_avg_shifts = 0;

    if (expected_insert_frac == 0) {
      ExpectedSearchIterationsAccumulator acc;
      build_node_implicit_from_existing(node, left, right, num_actual_keys, data_capacity, &acc, &model);
      expected_avg_exp_search_iterations = acc.get_stat();
    } else {
      ExpectedIterationsAndShiftsAccumulator acc(data_capacity);
      build_node_implicit_from_existing(node, left, right, num_actual_keys, data_capacity, &acc, &model);
      expected_avg_exp_search_iterations = acc.get_expected_num_search_iterations();
      expected_avg_shifts = acc.get_expected_num_shifts();
    }
    cost = kExpSearchIterationsWeight * expected_avg_exp_search_iterations +
           kShiftsWeight * expected_avg_shifts * expected_insert_frac;

    if (stats) {
      stats->num_search_iterations = expected_avg_exp_search_iterations;
      stats->num_shifts = expected_avg_shifts;
    }

    return cost;
  }

  // Helper function for compute_expected_cost
  // Implicitly build the data node in order to collect the stats
  static void build_node_implicit_from_existing(const self_type *node, int left,
                                                int right, int num_actual_keys,
                                                int data_capacity,
                                                StatAccumulator *acc,
                                                const LinearModel<T> *model) {
    int last_position = -1;
    int keys_remaining = num_actual_keys;
    const_iterator_type it(node, left);

    for (; it.cur_idx_ < right && !it.is_end(); it++) {
      int predicted_position = std::max(0, std::min(data_capacity - 1, model->predict(it.key())));
      int actual_position = std::max<int>(predicted_position, last_position + 1);
      int positions_remaining = data_capacity - actual_position;

      if (positions_remaining < keys_remaining) {
        actual_position = data_capacity - keys_remaining;
        for (; actual_position < data_capacity; actual_position++, it++) {
          predicted_position = std::max(0, std::min(data_capacity - 1, model->predict(it.key())));
          acc->accumulate(actual_position, predicted_position);
        }
        break;
      }

      acc->accumulate(actual_position, predicted_position);
      last_position = actual_position;
      keys_remaining--;
    }
  }

  /*** Bulk loading and model building ***/

  // Initalize key/payload/bitmap arrays and relevant metadata
  void initialize(int num_keys, double density) {
    num_keys_ = num_keys;
    data_capacity_ = std::max(static_cast<int>(num_keys / density), num_keys);
    bitmap_size_ = static_cast<size_t>(data_capacity_ == 0 ? 0 : std::ceil(data_capacity_ / 64.));
    bitmap_ = new (bitmap_allocator().allocate(bitmap_size_)) uint64_t[bitmap_size_]();  // initialize to all false
    buffer_ = new (buffer_allocator().allocate(data_capacity_)) AlexDataBuffer *[data_capacity_];
  }

  // Assumes pretrained_model is trained on dense array of keys
  void bulk_load(const V values[], int num_keys,
                 const LinearModel<T> *pretrained_model = nullptr,
                 bool train_with_sample = false) {
    initialize(num_keys, kInitDensity);

    if (num_keys == 0) {
      expansion_threshold_ = data_capacity_;
      contraction_threshold_ = 0;
      return;
    }

    // Build model
    if (pretrained_model != nullptr) {
      this->model_.a_ = pretrained_model->a_;
      this->model_.b_ = pretrained_model->b_;
    } else {
      build_model(values, num_keys, &(this->model_), train_with_sample);
    }
    this->model_.expand(static_cast<double>(data_capacity_) / num_keys);

    // Model-based inserts
    int last_position = -1;

    for (int i = 0; i < num_keys; i++) {
      int position = this->model_.predict(values[i].first);
      position = std::max<int>(position, last_position + 1);

      if (position >= data_capacity_) {
        continue;
      }

      if (check_exists(position)) {
        buffer_[position]->append(values[i].first, values[i].second);
      } else {
        buffer_[position] = new AlexDataBuffer(this, values[i].first, values[i].second);
        set_bit(position);
      }

      last_position = position;
    }

    expansion_threshold_ = std::min(std::max(data_capacity_ * kMaxDensity, static_cast<double>(num_keys + 1)),
                                    static_cast<double>(data_capacity_));
    contraction_threshold_ = data_capacity_ * kMinDensity;
    min_key_ = values[0].first;
    max_key_ = values[num_keys - 1].first;
  }

  // Bulk load using the keys between the left and right positions in
  // key/data_slots of an existing data node
  // keep_left and keep_right are set if the existing node was append-mostly
  // If the linear model and num_actual_keys have been precomputed, we can avoid
  // redundant work
  void bulk_load_from_existing(
      const self_type *node, int left, int right,
      bool keep_left = false, bool keep_right = false,
      const LinearModel<T> *precomputed_model = nullptr,
      int precomputed_num_actual_keys = -1) {
    assert(left >= 0 && right <= node->data_capacity_);

    std::cout << "bulk load from existing" << std::endl;

    // Build model
    int num_actual_keys = 0;
    if (precomputed_model == nullptr || precomputed_num_actual_keys == -1) {
      const_iterator_type it(node, left);
      LinearModelBuilder<T> builder(&(this->model_));

      int result = build_new(builder, it, 0, right);
      std::cout << "new_num_actual_keys: " << result << std::endl;
      num_actual_keys += result;
    } else {
      num_actual_keys = precomputed_num_actual_keys;
      this->model_.a_ = precomputed_model->a_;
      this->model_.b_ = precomputed_model->b_;
    }

    initialize(num_actual_keys, kMinDensity);

    if (num_actual_keys == 0) {
      expansion_threshold_ = data_capacity_;
      contraction_threshold_ = 0;
      return;
    }

    // Special casing if existing node was append-mostly
    if (keep_left) {
      this->model_.expand((num_actual_keys / kMaxDensity) / num_keys_);
    } else if (keep_right) {
      this->model_.expand((num_actual_keys / kMaxDensity) / num_keys_);
      this->model_.b_ += (data_capacity_ - (num_actual_keys / kMaxDensity));
    } else {
      this->model_.expand(static_cast<double>(data_capacity_) / num_keys_);
    }

    // Model-based inserts
    int last_position = -1;
    int keys_remaining = num_keys_;
    const_iterator_type it(node, left);

    for (; it.cur_idx_ < right && !it.is_end(); it++) {
      auto buf = it.buffer();
      auto keys = buf->get_keys();

      for (int i = 0; i < buf->size(); i++) {
        T key = keys[i];
        int position = this->model_.predict(key);
        position = std::max<int>(position, last_position + 1);
        last_position = std::max(last_position, position);

        if (check_exists(position)) {
          buffer_[position]->append(keys[i], *(buf->get_payload(i)));
        } else {
          buffer_[position] = new AlexDataBuffer(this, keys[i], *(buf->get_payload(i)));
          set_bit(position);
        }
      }
    }

    min_key_ = node->min_key_;
    max_key_ = node->max_key_;

    expansion_threshold_ =
      std::min(std::max(data_capacity_ * kMaxDensity,
                        static_cast<double>(num_keys_ + 1)),
                static_cast<double>(data_capacity_));
    contraction_threshold_ = data_capacity_ * kMinDensity;
  }

  static void build_model(const V *values, int num_keys, LinearModel<T> *model,
                          bool use_sampling = false) {
    if (use_sampling) {
      build_model_sampling(values, num_keys, model);
      return;
    }

    LinearModelBuilder<T> builder(model);
    for (int i = 0; i < num_keys; i++) {
      builder.add(values[i].first, i);
    }
    builder.build();
  }

  // Uses progressive non-random uniform sampling to build the model
  // Progressively increases sample size until model parameters are relatively
  // stable
  static void build_model_sampling(const V *values, int num_keys,
                                   LinearModel<T> *model,
                                   bool verbose = false) {
    const static int sample_size_lower_bound = 10;
    // If slope and intercept change by less than this much between samples,
    // return
    const static double rel_change_threshold = 0.01;
    // If intercept changes by less than this much between samples, return
    const static double abs_change_threshold = 0.5;
    // Increase sample size by this many times each iteration
    const static int sample_size_multiplier = 2;

    // If the number of keys is sufficiently small, we do not sample
    if (num_keys <= sample_size_lower_bound * sample_size_multiplier) {
      build_model(values, num_keys, model, false);
      return;
    }

    int step_size = 1;
    double sample_size = num_keys;
    while (sample_size >= sample_size_lower_bound) {
      sample_size /= sample_size_multiplier;
      step_size *= sample_size_multiplier;
    }
    step_size /= sample_size_multiplier;

    // Run with initial step size
    LinearModelBuilder<T> builder(model);
    for (int i = 0; i < num_keys; i += step_size) {
      builder.add(values[i].first, i);
    }
    builder.build();

    double prev_a = model->a_;
    double prev_b = model->b_;

    if (verbose) {
      std::cout << "Build index, sample size: " << num_keys / step_size
                << " (a, b): (" << prev_a << ", " << prev_b << ")" << std::endl;
    }

    // Keep decreasing step size (increasing sample size) until model does not
    // change significantly
    while (step_size > 1) {
      step_size /= sample_size_multiplier;
      // Need to avoid processing keys we already processed in previous samples
      int i = 0;
      while (i < num_keys) {
        i += step_size;
        for (int j = 1; (j < sample_size_multiplier) && (i < num_keys); j++, i += step_size) {
          builder.add(values[i].first, i);
        }
      }
      builder.build();

      double rel_change_in_a = std::abs((model->a_ - prev_a) / prev_a);
      double abs_change_in_b = std::abs(model->b_ - prev_b);
      double rel_change_in_b = std::abs(abs_change_in_b / prev_b);

      if (verbose) {
        std::cout << "Build index, sample size: " << num_keys / step_size
                  << " (a, b): (" << model->a_ << ", " << model->b_ << ") ("
                  << rel_change_in_a << ", " << rel_change_in_b << ")"
                  << std::endl;
      }
      if (rel_change_in_a < rel_change_threshold &&
          (rel_change_in_b < rel_change_threshold ||
           abs_change_in_b < abs_change_threshold)) {
        return;
      }
      prev_a = model->a_;
      prev_b = model->b_;
    }
  }

  // Unused function: builds a spline model by connecting the smallest and
  // largest points instead of using
  // a linear regression
  static void build_spline(const V *values, int num_keys, const LinearModel<T> *model) {
    int y_max = num_keys - 1;
    int y_min = 0;
    model->a_ = static_cast<double>(y_max - y_min) / (values[y_max].first - values[y_min].first);
    model->b_ = -1.0 * values[y_min].first * model->a_;
  }

  /*** Lookup ***/

  // Predicts the position of a key using the model
  inline int predict_position(const T &key) const {
    int position = this->model_.predict(key);
    position = std::max<int>(std::min<int>(position, data_capacity_ - 1), 0);
    return position;
  }

  // Searches for the last non-gap position equal to key
  // If no positions equal to key, returns -1
  /// If position deleted(not compressed), returns -2 
  int find_key(const T &key) {
    num_lookups_++;
    int position = predict_position(key);

    if (!check_exists(position)) {
      return -1;
    }
    
    auto buf = buffer_[position];
    return buf->find_idx(key);
  }

  int find_lower_bound(const T &key) {
    int position = predict_position(key);
    auto buf = buffer_[position];
    order_t prev_order = buf->min_idx;
    order_t next_order = buf->order_slots_[prev_order];
    T next_key = buf->key_slots_[next_order];

    while (next_order != -1) {
      if (next_key >= key) {
        return prev_order;
      }
      prev_order = next_order;
      next_order = buf->order_slots_[next_order];
      next_key = buf->key_slots_[next_order];
    }

    return -1;
  }

  int find_prev_key(const T &key) {
    int position = predict_position(key);
    auto buf = buffer_[position];
    order_t prev_order = buf->min_idx;
    T prev_key = buf->key_slots_[prev_order];
    order_t next_order = buf->order_slots_[prev_order];
    T next_key = buf->key_slots_[next_order];

    while (next_key != key) {
      if (prev_order == -1 || next_order == -1) {
        return -1;
      }
      prev_order = next_order;
      prev_key = next_key;
      next_order = buf->order_slots_[next_order];
      next_key = buf->key_slots_[next_order];
    }

    if (buf->deleted_slots_[prev_order] == 1 || buf->deleted_slots_[next_order] == 1) {
      return -1;
    }

    return prev_key;
  }

  int find_next_key(const T &key) {
    int position = predict_position(key);
    auto buf = buffer_[position];

    if (buf->count == 0 ||
      key_less(key, buf->key_slots_[buf->min_idx]) ||
      key_less(buf->key_slots_[buf->max_idx], key)) {
      return -1;
    }

    order_t next_order = buf->min_idx;
    T next_key = buf->key_slots_[next_order];

    while (next_key != key) {
      if (next_order == -1) {
        return -1;
      }
      next_order = buf->order_slots_[next_order];
      next_key = buf->key_slots_[next_order];
    };

    next_order = buf->order_slots_[next_order];
    next_key = buf->key_slots_[next_order];

    if (next_order == -1 || buf->deleted_slots_[next_order] == 1) {
      return -1;
    }

    return next_key;
  }

  // Starting from a position, return the first position that is not a gap
  // If no more filled positions, will return data_capacity
  // If exclusive is true, output is at least (pos + 1)
  // If exclusive is false, output can be pos itself
  int get_next_filled_buffer(int pos, bool exclusive = false) const {
    if (exclusive) {
      pos++;
      if (pos == data_capacity_) {
        return data_capacity_;
      }
    }

    int cur_bitmap_idx = pos >> 6;
    uint64_t cur_bitmap_data = bitmap_[cur_bitmap_idx];

    // Zero out extra bits
    int bit_pos = pos - (cur_bitmap_idx << 6);
    cur_bitmap_data &= ~((1ULL << (bit_pos)) - 1);

    while (cur_bitmap_data == 0) {
      cur_bitmap_idx++;
      if (cur_bitmap_idx >= bitmap_size_) {
        return data_capacity_;
      }
      cur_bitmap_data = bitmap_[cur_bitmap_idx];
    }

    uint64_t bit = extract_rightmost_one(cur_bitmap_data);
    return get_offset(cur_bitmap_idx, bit);
  }

  /*** Inserts and resizes ***/

  bool has_max_capacity() const {
    const_iterator_type it(this, 0);
    for (; it.cur_idx_ < data_capacity_ && !it.is_end(); it++) {
      if (!check_exists(it.cur_idx_)) {
        continue;
      }
      if (it->node_->is_totally_full()) {
        return true;
      }
    }
  }

  // Whether empirical cost deviates significantly from expected cost
  // Also returns false if empirical cost is sufficiently low and is not worth
  // splitting
  inline bool significant_cost_deviation() const {
    double emp_cost = empirical_cost();
    return emp_cost > kNodeLookupsWeight && emp_cost > 1.5 * this->cost_;
  }

  // Returns true if cost is catastrophically high and we want to force a split
  // The heuristic for this is if the number of shifts per insert (expected or
  // empirical) is over 100
  inline bool catastrophic_cost() const {
    return shifts_per_insert() > 100 || expected_avg_shifts_ > 100;
  }

  // First value in returned pair is fail flag:
  // 0 if successful insert (possibly with automatic expansion).
  // 1 if no insert because of significant cost deviation.
  // 2 if no insert because of "catastrophic" cost.
  // 3 if no insert because node is at max capacity.
  // -1 if key already exists and duplicates not allowed.
  //
  // Second value in returned pair is position of inserted key, or of the
  // already-existing key.
  // -1 if no insertion.
  std::pair<int, int> insert(const T &key, const P &payload) {
    // Insert
    int position = predict_position(key);
    std::cout << "position: " << position << std::endl;

    if (position < data_capacity_) {
      if (check_exists(position)) {
        auto buf = buffer_[position];
        if (buf->key_exists(key)) {
          std::cout << "key exists" << std::endl;
          return {-1, position};
        }

        bool result = buf->append(key, payload);
        if (!result) { /// try to insert into full buffer
          bool keep_left = is_append_mostly_right();
          bool keep_right = is_append_mostly_left();
          std::cout << "keep_left: " << keep_left << ", keep_right: " << keep_right << std::endl;
          
          result = resize(kMinDensity, true, keep_left, keep_right); /// retrain model
          if (!result) {
            return {3, position};
          }
          num_resizes_++;
          insert(key, payload); /// insert again
        }
      } else { /// first access at this position
        std::cout << "first" << std::endl;
        buffer_[position] = new AlexDataBuffer(this, key, payload);
        set_bit(position);
      }
    } else {
      return {3, -1};
    }

    // Update stats
    num_keys_++;
    num_inserts_++;
    if (key > max_key_) {
      max_key_ = key;
      num_right_out_of_bounds_inserts_++;
    }
    if (key < min_key_) {
      min_key_ = key;
      num_left_out_of_bounds_inserts_++;
    }
    return {0, position};
  }

  int build_new(LinearModelBuilder<T> &builder, const_iterator_type &it, int left, int right) {
    int c = 0;
    for (int i = left; it.cur_idx_ < right && !it.is_end(); it++) {
      std::cout << "build_new: " << it.cur_idx_ << std::endl;
      std::cout << "data_capacity_: " << it.node_->data_capacity_ << ", num_keys_: " << it.node_->num_keys_ << std::endl;
      std::cout << "min_key_: " << it.node_->min_key_ << ", max_key_: " << it.node_->max_key_ << std::endl;
      std::cout << "cur_idx_: " << it.cur_idx_ << ", cur_bitmap_idx_: " << it.cur_bitmap_idx_ << std::endl;
      auto buf = it.node_->buffer_[it.cur_bitmap_idx_];
      if (buf == nullptr) {
        printf("nullptr\n");
        continue;
      }
      printf("buf->min_idx: %d, buf->max_idx: %d\n", buf->min_idx, buf->max_idx);
      order_t next_order = buf->order_slots_[buf->min_idx];
      T next_key = buf->key_slots_[next_order];

      while (next_order != -1) {
        builder.add(next_key, i);
        next_order = buf->order_slots_[next_order];
        next_key = buf->key_slots_[next_order];
        i++;
        c++;
      }
    }
    builder.build();
    std::cout << "build_new end" << std::endl;
    return c;
  }

  // Resize the data node to the target density
  bool resize(double target_density, bool force_retrain = false,
              bool keep_left = false, bool keep_right = false) {
    if (num_keys_ == 0) {
      return false;
    }

    printf("resize\n");
    int new_data_capacity = std::max(static_cast<int>(num_keys_ / target_density), num_keys_);
    auto new_bitmap_size = static_cast<size_t>(std::ceil(new_data_capacity / 64.));
    auto new_bitmap = new (bitmap_allocator().allocate(new_bitmap_size)) uint64_t[new_bitmap_size]();  // initialize to all false
    auto new_buffer = new (buffer_allocator().allocate(new_data_capacity)) AlexDataBuffer *[new_data_capacity];

    // Retrain model if the number of keys is sufficiently small (under 50)
    if (num_keys_ < 50 || force_retrain) {
      const_iterator_type it(this, 0);
      LinearModelBuilder<T> builder(&(this->model_));

      build_new(builder, it, 0, data_capacity_);

      if (keep_left) {
        this->model_.expand(static_cast<double>(data_capacity_) / num_keys_);
      } else if (keep_right) {
        this->model_.expand(static_cast<double>(data_capacity_) / num_keys_);
        this->model_.b_ += (new_data_capacity - data_capacity_);
      } else {
        this->model_.expand(static_cast<double>(new_data_capacity) / num_keys_);
      }
    } else {
      if (keep_right) {
        this->model_.b_ += (new_data_capacity - data_capacity_);
      } else if (!keep_left) {
        this->model_.expand(static_cast<double>(new_data_capacity) / data_capacity_);
      }
    }

    int last_position = -1;

    const_iterator_type it(this, 0);
    for (; it.cur_idx_ < data_capacity_ && !it.is_end(); it++) {
      auto buf = it.buffer();
      auto keys = buf->get_keys();

      for (int i = 0; i < buf->size(); i++) {
        T key = keys[i];
        int position = this->model_.predict(key);
        position = std::max<int>(position, last_position + 1);

        if (position < new_data_capacity) {
          int new_bitmap_pos = position >> 6;
          int new_bit_pos = position - (new_bitmap_pos << 6);
          bool new_check_exists = new_bitmap[new_bitmap_pos] & (1ULL << new_bit_pos);

          if (new_check_exists) {
            auto payload = *(buf->get_payload(i));
            new_buffer[position]->append(keys[i], *(buf->get_payload(i)));
          } else {
            new_buffer[position] = new AlexDataBuffer(this, keys[i], *(buf->get_payload(i)));
            set_bit(new_bitmap, position);
          }
        }

        last_position = position;
      }
    }

    bitmap_allocator().deallocate(bitmap_, bitmap_size_);
    buffer_allocator().deallocate(buffer_, data_capacity_);

    data_capacity_ = new_data_capacity;
    bitmap_size_ = new_bitmap_size;
    bitmap_ = new_bitmap;
    buffer_ = new_buffer;
    expansion_threshold_ = std::min(std::max(data_capacity_ * kMaxDensity, static_cast<double>(num_keys_ + 1)), static_cast<double>(data_capacity_));
    contraction_threshold_ = data_capacity_ * kMinDensity;

    return true;
  }

  inline bool is_append_mostly_right() const {
    return static_cast<double>(num_right_out_of_bounds_inserts_) / num_inserts_ > kAppendMostlyThreshold;
  }

  inline bool is_append_mostly_left() const {
    return static_cast<double>(num_left_out_of_bounds_inserts_) / num_inserts_ > kAppendMostlyThreshold;
  }

  /*** Deletes ***/

  // Erase key with the input value
  // Returns whether key is erased (there cannot be multiple keys with the same
  // value)
  bool erase(const T &key) {
    int position = predict_position(key);
    if (position < 0 || position >= data_capacity_ || !check_exists(position)) return false;

    int pos = find_key(key);
    if (pos < 0) return false;

    auto buf = buffer_[position];
    buf->delete_key(pos);

    num_keys_--;
    if (num_keys_ < contraction_threshold_) {
      resize(kMaxDensity);  // contract
      num_resizes_++;
    }
    return true;
  }

  // Erase keys with value between start key (inclusive) and end key.
  // Returns the number of keys erased.
  int erase_range(T start_key, T end_key, bool end_key_inclusive = false) {
    int position = predict_position(start_key);
    int start_pos = find_key(start_key);
    if (start_pos < 0) return 0;

    int num_erased = 0;
    auto buf = buffer_[position];
    order_t next_order = buf->order_slots_[start_pos];
    T next_key = buf->key_slots_[next_order];

    while (next_key != end_key) {
      if (next_order == -1) {
        position = get_next_filled_buffer(position, true);
        buf = buffer_[position];
        next_order = buf->min_idx;
        next_key = buf->key_slots_[next_order];
        continue;
      }

      buf->delete_key(next_order);
      num_erased++;
      next_order = buf->order_slots_[next_order];
      next_key = buf->key_slots_[next_order];
    }

    num_keys_ -= num_erased;

    if (num_keys_ < contraction_threshold_) {
      resize(kMaxDensity); // contract
      num_resizes_++;
    }
    return num_erased;
  }

  /*** Stats ***/

  // Total size of node metadata
  long long node_size() const override { return sizeof(self_type); }

  // Total size in bytes of key/payload/data_slots and bitmap
  long long data_size() const {
    long long data_size = data_capacity_ * sizeof(T);
    data_size += data_capacity_ * sizeof(P);
    data_size += bitmap_size_ * sizeof(uint64_t);
    return data_size;
  }

  // Number of contiguous blocks of keys without gaps
  int num_packed_regions() const {
    int num_packed = 0;
    bool is_packed = check_exists(0);

    for (int i = 1; i < data_capacity_; i++) {
      if (check_exists(i) != is_packed) {
        if (is_packed) {
          num_packed++;
        }
        is_packed = !is_packed;
      }
    }
    if (is_packed) {
      num_packed++;
    }
    return num_packed;
  }

  /*** Debugging ***/

  bool validate_structure(bool verbose = false) const {
    if (this->cost_ < 0 || std::isnan(this->cost_)) {
      std::cout << "[Data node cost is invalid value]"
                << " node addr: " << this << ", node level: " << this->level_
                << ", cost: " << this->cost_ << std::endl;
      return false;
    }

    for (int i = 0; i < data_capacity_ - 1; i++) {
      if (buffer_[i] == nullptr) continue;

      int next_pos = get_next_filled_buffer(i, true);

      if (key_greater(buffer_[i]->get_key(), buffer_[next_pos]->get_key())) {
        if (verbose) {
          std::cout << "Keys should be in non-increasing order" << std::endl;
        }
        return false;
      } else if (key_less(buffer_[i]->get_key(), buffer_[next_pos]->get_key()) &&
                 !check_exists(i)) {
        if (verbose) {
          std::cout << "The last key of a certain value should not be a gap"
                    << std::endl;
        }
        return false;
      }
    }

    uint64_t num_bitmap_ones = 0;
    for (int i = 0; i < bitmap_size_; i++) {
      num_bitmap_ones += count_ones(bitmap_[i]);
    }

    if (static_cast<int>(num_bitmap_ones) != num_keys_) {
      if (verbose) {
        std::cout << "Number of ones in bitmap should match num_keys"
                  << std::endl;
      }
      return false;
    }
    return true;
  }

  std::string to_string() const {
    std::string str;
    str += "Num keys: " + std::to_string(num_keys_) + ", Capacity: " +
           std::to_string(data_capacity_) + ", Expansion Threshold: " +
           std::to_string(expansion_threshold_) + "\n";
    return str;
  }

  void print_buffer() {
    for (int i = 0; i < data_capacity_; i++) {
      if (check_exists(i)) {
        std::cout << "position: " << i << " => ";
        buffer_[i]->print();
        std::cout << std::endl;
      }
    }
  }
};
}