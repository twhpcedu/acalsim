/*
 * Copyright 2023-2026 Playlab/ACAL
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file Template.cc
 * @brief Implementation file for Template simulator component
 *
 * This file provides implementation details for the Template class declared in Template.hh.
 * Since Template uses inline method implementations in the header file (common for template-
 * based designs and simple components), this file serves primarily as a placeholder for:
 *
 * 1. **Complex Method Implementations:**
 *    - Move large method bodies from header to here to reduce compilation time
 *    - Implement private helper methods
 *    - Implement static member initializations
 *
 * 2. **Implementation-Specific Includes:**
 *    - Include additional headers not needed by users of Template.hh
 *    - Avoid polluting header namespace with implementation details
 *
 * 3. **Project Organization:**
 *    - Maintain consistent file structure across simulator components
 *    - Separate interface (Template.hh) from implementation (Template.cc)
 *
 * **Best Practices for Implementation Files:**
 *
 * - Keep header file clean: Only declarations and inline simple methods
 * - Move complex logic here: Large functions, helper methods, implementation details
 * - Use anonymous namespaces for file-local helpers:
 *   ```cpp
 *   namespace {
 *       // Helper functions only visible in this .cc file
 *       void helperFunction() { ... }
 *   }
 *   ```
 *
 * - Separate concerns: Group related functionality with comments
 * - Document non-obvious implementations with inline comments
 *
 * **Example Implementation Pattern:**
 * ```cpp
 * #include "Template.hh"
 * #include "detailed_impl_headers.hh"  // Not in public header
 *
 * namespace acalsim {
 *
 * // Static member initialization
 * int Template::static_counter = 0;
 *
 * // Complex method implementation
 * void Template::complexMethod() {
 *     // Implementation details...
 * }
 *
 * // Private helper methods
 * void Template::privateHelper() {
 *     // Only declared in private section of header
 * }
 *
 * }  // namespace acalsim
 * ```
 *
 * @see Template.hh For class interface and usage documentation
 */

#include "Template.hh"

/**
 * @note Template class uses inline implementations in Template.hh.
 *       This file is provided for future expansion and implementation-specific code.
 *
 * **To add implementation code:**
 * 1. Declare method in Template.hh (without inline implementation)
 * 2. Implement method here in Template.cc
 * 3. Include necessary implementation-specific headers
 * 4. Keep header clean and focused on interface
 *
 * **Example:**
 * ```cpp
 * // In Template.hh:
 * class Template : public CPPSimBase {
 *     void complexInitialization();  // Declaration only
 * };
 *
 * // In Template.cc:
 * void Template::complexInitialization() {
 *     // Complex implementation logic here
 *     // Can use headers not included in Template.hh
 * }
 * ```
 */
