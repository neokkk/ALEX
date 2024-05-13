// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*
 * This short sample program demonstrates ALEX's API.
 */

#include "../core/alex.h"

#define KEY_TYPE int
#define PAYLOAD_TYPE int

int main(int, char **) {
  // Create some synthetic data: keys are dense integers between 0 and 99, and
  // payloads are random values
  const int num_keys = 100;
  std::pair<KEY_TYPE, PAYLOAD_TYPE> values[num_keys];
  std::mt19937_64 gen(std::random_device{}());
  std::uniform_int_distribution<PAYLOAD_TYPE> dis;
  
  for (int i = 0; i < num_keys; i++) {
    values[i].first = i;
    values[i].second = dis(gen);
  }

  alex::Alex<KEY_TYPE, PAYLOAD_TYPE> index;

  // Bulk load the keys [0, 100)
  index.bulk_load(values, num_keys);

  // Insert the keys [100, 200). Now there are 200 keys.
  for (int i = num_keys; i < 2 * num_keys; i++) {
    KEY_TYPE new_key = i;
    PAYLOAD_TYPE new_payload = dis(gen);
    index.insert(new_key, new_payload);
  }

  // Erase the keys [0, 10). Now there are 190 keys.
  for (int i = 0; i < 10; i++) {
    int result = index.erase(i);
  }

  // Iterate through all entries in the index and update their payload if the
  // key is even
  // int num_entries = 0;
  // for (auto it = index.begin(); it != index.end(); it++) {
  //   if (it.key() % 2 == 0) {  // it.key() is equivalent to (*it).first
  //     it.payload() = dis(gen);
  //   }
  //   num_entries++;
  // }
  // if (num_entries != 190) {
  //   std::cout << "Error! There should be 190 entries in the index."
  //             << std::endl;
  // }

  // // Insert 9 more keys with value 42. Now there are 199 keys.
  for (int i = 0; i < 9; i++) {
    KEY_TYPE new_key = 42;
    PAYLOAD_TYPE new_payload = dis(gen);
    index.insert(new_key, new_payload);
  }

  // Look at some stats
  auto stats = index.get_stats();
  stats.print();
}