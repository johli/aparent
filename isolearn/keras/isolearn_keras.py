from __future__ import print_function
import keras

import pandas as pd
import numpy as np
import scipy.sparse as sp


class BatchEncoder :
    
    def __init__(self, encoder) :
        self.encoder = encoder
    
    def encode(self, seqs) :
        
        batch_dims = tuple([len(seqs)] + list(self.encoder.encode_dims))
        encodings = np.zeros(batch_dims)
        
        self.encode_inplace(seqs, encodings)
        
        return encodings
    
    def encode_inplace(self, seqs, encodings) :
        for i in range(0, len(seqs)) :
            self.encoder.encode_inplace(seqs[i], encodings[i,])
    
    def encode_row_sparse(self, seqs) :
        return sp.csr_matrix(self.encode_sparse(seqs))
    
    def encode_col_sparse(self, seqs) :
        return sp.csc_matrix(self.encode_sparse(seqs))
    
    def encode_sparse(self, seqs) :
        n_cols = np.prod(np.ravel(list(self.encoder.encode_dims)))
        encoding_mat = sp.lil_matrix((len(seqs), n_cols))
        for i in range(0, len(seqs)) :
            self.encoder.encode_inplace_sparse(seqs[i], encoding_mat, i)
        
        return encoding_mat
    
    def decode(self, encodings) :
        decodings = []
        for i in range(0, encodings.shape[0]) :
            decodings.append(self.encoder.decode(encodings[i,]))
        
        return decodings
    
    def decode_sparse(self, encoding_mat) :
        decodings = []
        for i in range(0, encoding_mat.shape[0]) :
            decodings.append(self.encoder.decode_sparse(encoding_mat, i))
        
        return decodings
    
    def __call__(self, seqs) :
        return self.encode(seqs)

class SparseBatchEncoder(BatchEncoder) :
    
    def __init__(self, encoder, sparse_mode='row') :
        super(SparseBatchEncoder, self).__init__(encoder)
        
        self.sparse_mode = sparse_mode
    
    def encode(self, seqs) :
        return self.__call__(seqs)
    
    def decode(self, encodings) :
        return self.decode_sparse(encodings)
    
    def __call__(self, seqs) :
        if self.sparse_mode == 'row' :
            return self.encode_row_sparse(seqs)
        elif self.sparse_mode == 'col' :
            return self.encode_col_sparse(seqs)
        else :
            return self.encode_sparse(seqs)


    
class BatchTransformer :
    
    def __init__(self, transformer) :
        self.transformer = transformer
    
    def transform(self, values) :
        
        batch_dims = tuple([values.shape[0]] + list(self.transformer.transform_dims))
        transforms = np.zeros(batch_dims)
        
        self.transform_inplace(values, transforms)
        
        return transforms
    
    def transform_inplace(self, values, transforms) :
        for i in range(0, values.shape[0]) :
            self.transformer.transform_inplace(values[i], transforms[i,])
    
    def transform_row_sparse(self, values) :
        return sp.csr_matrix(self.transform_sparse(values))
    
    def transform_col_sparse(self, values) :
        return sp.csc_matrix(self.transform_sparse(values))
    
    def transform_sparse(self, values) :
        n_cols = np.prod(np.ravel(list(self.transformer.transform_dims)))
        transform_mat = sp.lil_matrix((values.shape[0], n_cols))
        for i in range(0, values.shape[0]) :
            self.transformer.transform_inplace_sparse(values[i], transform_mat, i)
        
        return transform_mat
    
    def __call__(self, values) :
        return self.transform(values)

class SparseBatchTransformer(BatchTransformer) :
    
    def __init__(self, transformer, sparse_mode='row') :
        super(SparseBatchTransformer, self).__init__(transformer)
        
        self.sparse_mode = sparse_mode
    
    def transform(self, values) :
        return self.__call__(values)
    
    def __call__(self, values) :
        if self.sparse_mode == 'row' :
            return self.transform_row_sparse(values)
        elif self.sparse_mode == 'col' :
            return self.transform_col_sparse(values)
        else :
            return self.transform_sparse(values)


class ValueTransformer :
    
    def __init__(self, transformer_type_id, transform_dims) :
        self.transformer_type_id = transformer_type_id
        self.transform_dims = transform_dims
    
    def transform(self, values) :
        raise NotImplementedError()
    
    def transform_inplace(self, values, transform) :
        raise NotImplementedError()
    
    def transform_inplace_sparse(self, values, transform_mat, row_index) :
        raise NotImplementedError()
    
    def __call__(self, values) :
        return self.transform(values)


class LogitTransformer(ValueTransformer) :
    
    def __init__(self, n_classes) :
        super(LogitTransformer, self).__init__('logit', (n_classes, ))
        
        self.n_classes = n_classes
    
    def transform(self, values) :
        logits = np.zeros(self.n_classes)
        self.transform_inplace(values, logits)

        return logits
    
    def transform_inplace(self, values, transform) :
        transform[:] = np.log2(values / (1. - values))
    
    def transform_inplace_sparse(self, values, transform_mat, row_index) :
        logits = self.transform(values)
        transform_mat[row_index, :] = np.ravel(logits)

class CountSumTransformer(ValueTransformer) :
    
    def __init__(self, sparse_source=False) :
        super(CountSumTransformer, self).__init__('count', (1, ))
        
        self.sparse_source = sparse_source
    
    def transform(self, values) :
        c = np.zeros(1)
        self.transform_inplace(values, c)

        return c
    
    def transform_inplace(self, values, transform) :
        if not self.sparse_source :
            transform[0] = np.sum(values)
        else :
            transform[0] = values[0, :].sum(axis=-1)
    
    def transform_inplace_sparse(self, values, transform_mat, row_index) :
        c = self.transform(values)
        transform_mat[row_index, :] = np.ravel(c)

class ProportionTransformer(ValueTransformer) :
    
    def __init__(self, isoform_range=[0], normalizer_range=[0, 1], laplace_smoothing=0.001, sparse_source=False) :
        super(ProportionTransformer, self).__init__('proportion', (1, ))
        
        self.isoform_range = isoform_range
        self.normalizer_range = normalizer_range
        self.laplace_smoothing = laplace_smoothing
        self.sparse_source = sparse_source
        
        self.isoform_len = float(len(isoform_range))
        self.normalizer_len = float(len(normalizer_range))
    
    def transform(self, values) :
        prop = np.zeros(1)
        self.transform_inplace(values, prop)

        return prop
    
    def transform_inplace(self, values, transform) :
        if not self.sparse_source :
            transform[0] = np.sum(values[self.isoform_range] + self.laplace_smoothing) / np.sum(values[self.normalizer_range] + self.laplace_smoothing)
        else :
            transform[0] = (values[0, self.isoform_range].sum(axis=-1) + self.isoform_len * self.laplace_smoothing) / (values[0, self.normalizer_range].sum(axis=-1) + self.normalizer_len * self.laplace_smoothing)
    
    def transform_inplace_sparse(self, values, transform_mat, row_index) :
        prop = self.transform(values)
        transform_mat[row_index, :] = np.ravel(prop)
    
class MultiProportionTransformer(ValueTransformer) :
    
    def __init__(self, n_classes=2, laplace_smoothing=0.0001, sparse_source=False) :
        super(MultiProportionTransformer, self).__init__('multi_proportion', (n_classes, ))
        
        self.n_classes = n_classes
        self.laplace_smoothing = laplace_smoothing
        self.sparse_source = sparse_source
        
        self.vector_len = float(n_classes)
    
    def transform(self, values) :
        prop = np.zeros(self.n_classes)
        self.transform_inplace(values, prop)

        return prop
    
    def transform_inplace(self, values, transform) :
        if self.sparse_source :
            values = np.ravel(values.todense())
        
        transform[:] = (values + self.laplace_smoothing) / np.sum(values + self.laplace_smoothing)
    
    def transform_inplace_sparse(self, values, transform_mat, row_index) :
        prop = self.transform(values)
        transform_mat[row_index, :] = np.ravel(prop)






class SequenceEncoder :
    
    def __init__(self, encoder_type_id, encode_dims) :
        self.encoder_type_id = encoder_type_id
        self.encode_dims = encode_dims
    
    def encode(self, seq) :
        raise NotImplementedError()
    
    def encode_inplace(self, seq, encoding) :
        raise NotImplementedError()
    
    def encode_inplace_sparse(self, seq, encoding_mat, row_index) :
        raise NotImplementedError()
    
    def decode(self, encoding) :
        raise NotImplementedError()
    
    def decode_sparse(self, encoding_mat, row_index) :
        raise NotImplementedError()
    
    def __call__(self, seq) :
        return self.encode(seq)
    

class OneHotEncoder(SequenceEncoder) :
    
    def __init__(self, seq_length=100, default_fill_value=0) :
        super(OneHotEncoder, self).__init__('one_hot', (seq_length, 4))
        
        self.seq_length = seq_length
        self.default_fill_value = default_fill_value
        self.encode_map = {
            'A' : 0,
            'C' : 1,
            'G' : 2,
            'T' : 3
        }
        self.decode_map = {
                0 : 'A',
                1 : 'C',
                2 : 'G',
                3 : 'T',
                -1 : 'X'
        }
    
    def encode(self, seq) :
        one_hot = np.zeros((self.seq_length, 4))
        self.encode_inplace(seq, one_hot)

        return one_hot
    
    def encode_inplace(self, seq, encoding) :
        for pos, nt in enumerate(list(seq)) :
            if nt in self.encode_map :
                encoding[pos, self.encode_map[nt]] = 1
            elif self.default_fill_value != 0 :
                encoding[pos, :] = self.default_fill_value
    
    def encode_inplace_sparse(self, seq, encoding_mat, row_index) :
        encoding = self.encode(seq)
        encoding_mat[row_index, :] = np.ravel(encoding)
    
    def decode(self, encoding) :
        seq = ''
    
        for pos in range(0, encoding.shape[0]) :
            argmax_nt = np.argmax(encoding[pos, :])
            max_nt = np.max(encoding[pos, :])
            if max_nt == 1 :
                seq += self.decode_map[argmax_nt]
            else :
                seq += self.decode_map[-1]

        return seq
    
    def decode_sparse(self, encoding_mat, row_index) :
        encoding = np.array(encoding_mat[row_index, :].todense()).reshape(-1, 4)
        return self.decode(encoding)


class NMerEncoder(SequenceEncoder) :
    
    def __init__(self, n_mer_len=6, count_n_mers=True) :
        super(NMerEncoder, self).__init__('mer_' + str(n_mer_len), (4**n_mer_len, ))
        
        self.count_n_mers = count_n_mers
        self.n_mer_len = n_mer_len
        self.encode_order = ['A', 'C', 'G', 'T']
        self.n_mers = self._get_ordered_nmers(n_mer_len)
        
        self.encode_map = {
            n_mer : n_mer_index for n_mer_index, n_mer in enumerate(self.n_mers)
        }
        
        self.decode_map = {
            n_mer_index : n_mer for n_mer_index, n_mer in enumerate(self.n_mers)
        }
    
    def _get_ordered_nmers(self, n_mer_len) :
        
        if n_mer_len == 0 :
            return []
        
        if n_mer_len == 1 :
            return list(self.encode_order.copy())
        
        n_mers = []
        
        prev_n_mers = self._get_ordered_nmers(n_mer_len - 1)
        
        for _, prev_n_mer in enumerate(prev_n_mers) :
            for _, nt in enumerate(self.encode_order) :
                n_mers.append(prev_n_mer + nt)
        
        return n_mers
            
    def encode(self, seq) :
        n_mer_vec = np.zeros(self.n_mer_len)
        self.encode_inplace(seq, n_mer_vec)

        return n_mer_vec
    
    def encode_inplace(self, seq, encoding) :
        for i_start in range(0, len(seq) - self.n_mer_len + 1) :
            i_end = i_start + self.n_mer_len
            n_mer = seq[i_start:i_end]
            
            if n_mer in self.encode_map :
                if self.count_n_mers :
                    encoding[self.encode_map[n_mer]] += 1
                else :
                    encoding[self.encode_map[n_mer]] = 1
    
    def encode_inplace_sparse(self, seq, encoding_mat, row_index) :
        for i_start in range(0, len(seq) - self.n_mer_len + 1) :
            i_end = i_start + self.n_mer_len
            n_mer = seq[i_start:i_end]
            
            if n_mer in self.encode_map :
                if self.count_n_mers :
                    encoding_mat[row_index, self.encode_map[n_mer]] += 1
                else :
                    encoding_mat[row_index, self.encode_map[n_mer]] = 1
    
    def decode(self, encoding) :
        n_mers = {}
    
        for i in range(0, encoding.shape[0]) :
            if encoding[i] != 0 :
                n_mers[self.decode_map[i]] = encoding[i]

        return n_mers
    
    def decode_sparse(self, encoding_mat, row_index) :
        encoding = np.ravel(encoding_mat[row_index, :].todense())
        return self.decode(encoding)

class CategoricalEncoder(SequenceEncoder) :
    
    def __init__(self, n_categories=2, categories=['default_1', 'default_2'], category_index=None) :
        super(CategoricalEncoder, self).__init__('categorical', (n_categories, ))
        
        self.n_categories = n_categories
        self.categories = categories
        self.category_index = category_index
        if self.category_index is None :
            self.category_index = list(np.arange(n_categories, dtype=np.int).tolist())
        
        self.encode_map = {
            category : category_id for category_id, category in zip(self.category_index, self.categories)
        }
        
        self.decode_map = {
            category_id : category for category_id, category in zip(self.category_index, self.categories)
        }
            
    def encode(self, seq) :
        n_mer_vec = np.zeros(self.n_categories)
        self.encode_inplace(seq, n_mer_vec)

        return n_mer_vec
    
    def encode_inplace(self, seq, encoding) :
        encoding[self.encode_map[seq]] = 1
    
    def encode_inplace_sparse(self, seq, encoding_mat, row_index) :
        encoding_mat[row_index, self.encode_map[seq]] = 1
    
    def decode(self, encoding) :
        category = self.decode_map[np.argmax(encoding)]

        return category
    
    def decode_sparse(self, encoding_mat, row_index) :
        encoding = np.ravel(encoding_mat[row_index, :].todense())
        return self.decode(encoding)

class SequenceExtractor :
    
    def __init__(self, df_column, start_pos=0, end_pos=100, shifter=None) :
        self.df_column = df_column
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.shifter = shifter
    
    def extract(self, raw_input, index=None) :
        shift_pos = 0
        if self.shifter is not None :
            shift_pos = self.shifter.get_random_sample(index)
        
        return raw_input[self.df_column][self.start_pos + shift_pos: self.end_pos + shift_pos]
    
    def __call__(self, raw_input, index=None) :
        return self.extract(raw_input, index)

class CountExtractor :
    
    def __init__(self, df_column=None, start_pos=0, end_pos=100, static_poses=None, shifter=None, sparse_source=False) :
        self.df_column = df_column
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.shifter = shifter
        self.sparse_source = sparse_source
        self.static_poses = static_poses
    
    def extract(self, raw_input, index) :
        shift_pos = 0
        if self.shifter is not None :
            shift_pos = self.shifter.get_random_sample(index)
        
        dense_input = None
        if not self.sparse_source :
            dense_input = raw_input
        else :
            dense_input = np.ravel(raw_input.todense())
        
        if self.df_column is None :
            extracted_values = dense_input[self.start_pos + shift_pos: self.end_pos + shift_pos]
        else :
            extracted_values = dense_input[self.df_column][self.start_pos + shift_pos: self.end_pos + shift_pos]
        
        if self.static_poses is not None :
            if self.df_column is None :
                extracted_values = np.concatenate([extracted_values, dense_input[self.static_poses]], axis=0)
            else :
                extracted_values = np.concatenate([extracted_values, dense_input[self.df_column][self.static_poses]], axis=0)
        
        return extracted_values
    
    def __call__(self, raw_input, index=None) :
        return self.extract(raw_input, index)


class PositionShifter :
    
    def __init__(self, shift_range, shift_probs) :
        self.shift_range = shift_range
        self.shift_probs = shift_probs
        self.position_shift = 0
        
    def get_random_sample(self, index=None) :
        if index is None :
            return self.position_shift
        else :
            return self.position_shift[index]
    
    def generate_random_sample(self, batch_size=1, batch_indexes=None) :
        self.position_shift = np.random.choice(self.shift_range, size=batch_size, replace=True, p=self.shift_probs)

class CutAlignSampler(PositionShifter) :
    
    def __init__(self, cuts, window_size, align_pos, fixed_poses, p_fixed, p_pos, p_neg, sparse_source=False) :
        self.cuts = cuts
        self.window_size = window_size
        self.align_pos = align_pos
        self.sparse_source = sparse_source
        self.fixed_poses = fixed_poses
        self.p_fixed = p_fixed
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.position_shift = 0
        
        self.fixed_pos_mask = np.ones(cuts.shape[1])
        for j in range(len(fixed_poses)) :
            self.fixed_pos_mask[fixed_poses[j]] = 0
        
    def get_random_sample(self, index=None) :
        if index is None :
            return self.position_shift
        else :
            return self.position_shift[index]
    
    def generate_random_sample(self, batch_size=1, batch_indexes=None) :
        batch_cuts = self.cuts[batch_indexes, :]
        if self.sparse_source :
            batch_cuts = np.array(batch_cuts.todense())
        
        cut_pos = np.arange(batch_cuts.shape[1])
        
        nonzero_cuts = [
            np.nonzero( ((batch_cuts[i, :] > 0) & (self.fixed_pos_mask == 1)) & ((cut_pos >= int(self.window_size / 2)) & (cut_pos < batch_cuts.shape[1] - int(self.window_size / 2))) )[0] for i in range(batch_cuts.shape[0])
        ]
        zero_cuts = [
            np.nonzero( ((batch_cuts[i, :] == 0) & (self.fixed_pos_mask == 1)) & ((cut_pos >= int(self.window_size / 2)) & (cut_pos < batch_cuts.shape[1] - int(self.window_size / 2))) )[0] for i in range(batch_cuts.shape[0])
        ]
        
        sampled_shifts = np.zeros(batch_size, dtype=np.int)
        
        sample_mode_choices = np.random.choice([1, 2, 3], size=batch_size, replace=True, p=[self.p_fixed, self.p_pos, self.p_neg])
        for i in range(batch_cuts.shape[0]) :
            sampled_pos = self.align_pos
            if sample_mode_choices[i] == 1 or (len(nonzero_cuts[i]) == 0 and len(zero_cuts[i]) == 0) : #Fixed pos
                sampled_pos = np.random.choice(self.fixed_poses)
            elif sample_mode_choices[i] == 2 and len(nonzero_cuts[i]) > 0 : #Pos sample
                sampled_pos = np.random.choice(nonzero_cuts[i])
            elif sample_mode_choices[i] == 3 and len(zero_cuts[i]) > 0 : #Neg sample
                sampled_pos = np.random.choice(zero_cuts[i])
            sampled_shifts[i] = int(sampled_pos - self.align_pos)
        
        self.position_shift = sampled_shifts

class DataGenerator(keras.utils.Sequence) :
    
    def __init__(self, data_ids, sources, batch_size=32, inputs=None, outputs=None, randomizers=None, shuffle=True, densify_batch_matrices=False) :
        self.data_ids = data_ids
        self.sources = sources
        self.batch_size = batch_size
        self.inputs = inputs
        self.outputs = outputs
        self.randomizers = randomizers
        self.shuffle = shuffle
        self.densify_batch_matrices = densify_batch_matrices
        
        self._init_encoders()

        if isinstance(self.shuffle, DataGenerator) :
            self.indexes = self.shuffle.indexes
        else :
            self.indexes = np.arange(len(self.data_ids))

        self.on_epoch_end()
    
    def _init_encoders(self) :
        self.encoders = {}
        self.transformers = {}
        
        for input_dict in self.inputs :
            if 'sparse' not in input_dict or not input_dict['sparse'] :
                if 'encoder' in input_dict and input_dict['encoder'] is not None and isinstance(input_dict['encoder'], SequenceEncoder) :
                    input_dict['encoder'] = BatchEncoder(input_dict['encoder'])
                elif 'transformer' in input_dict and input_dict['transformer'] is not None and isinstance(input_dict['transformer'], ValueTransformer) :
                    input_dict['transformer'] = BatchTransformer(input_dict['transformer'])
            else :
                sparse_mode = 'row'
                if 'sparse_mode' in input_dict :
                    sparse_mode = input_dict['sparse_mode']
                if 'encoder' in input_dict and input_dict['encoder'] is not None and isinstance(input_dict['encoder'], SequenceEncoder) :
                    input_dict['encoder'] = SparseBatchEncoder(input_dict['encoder'], sparse_mode=sparse_mode)
                elif 'transformer' in input_dict and input_dict['transformer'] is not None and isinstance(input_dict['transformer'], ValueTransformer) :
                    input_dict['transformer'] = SparseBatchTransformer(input_dict['transformer'], sparse_mode=sparse_mode)
            
            if 'encoder' in input_dict and input_dict['encoder'] is not None :
                self.encoders[input_dict['id']] = input_dict['encoder']
            elif 'transformer' in input_dict and input_dict['transformer'] is not None :
                self.transformers[input_dict['id']] = input_dict['transformer']
        
            if 'encoder' not in input_dict :
                input_dict['encoder'] = None
            if 'transformer' not in input_dict :
                input_dict['transformer'] = None
        if self.outputs is not None :
            for output_dict in self.outputs :
                if output_dict['source_type'] != 'zeros' and ('transformer' in output_dict and output_dict['transformer'] is not None) :
                    if ('sparse' not in output_dict or not output_dict['sparse']) and isinstance(output_dict['transformer'], ValueTransformer) :
                        output_dict['transformer'] = BatchTransformer(output_dict['transformer'])
                    elif isinstance(output_dict['transformer'], ValueTransformer) :
                        sparse_mode = 'row'
                        if 'sparse_mode' in output_dict :
                            sparse_mode = output_dict['sparse_mode']
                        output_dict['transformer'] = SparseBatchTransformer(output_dict['transformer'], sparse_mode=sparse_mode)

                    self.transformers[output_dict['id']] = output_dict['transformer']
                else :
                    self.transformers[output_dict['id']] = None

    def __len__(self) :
        return int(np.floor(len(self.data_ids) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Generate random samples for current batch
        for randomizer in self.randomizers :
            randomizer.generate_random_sample(self.batch_size, self.data_ids[indexes])

        # Generate data
        batch_tuple = self._generate_batch(self.data_ids[indexes])

        return batch_tuple

    def on_epoch_end(self) :
        if not isinstance(self.shuffle, DataGenerator) and self.shuffle == True :
            np.random.shuffle(self.indexes)
        
        #for randomizer in self.randomizers :
        #    randomizer.generate_random_sample(self.batch_size)
    
    def _generate_batch(self, batch_indexes) :
        
        generated_inputs = []
        generated_outputs = None
        
        #Generate inputs
        for input_dict in self.inputs :
            input_source = self.sources[input_dict['source']]
            if input_dict['source_type'] == 'dataframe' :
                #source_batch = self.data.iloc[batch_indexes].apply(lambda row: input_dict['extractor'](row, ), axis=1).values
                
                #apply_func = _get_df_apply_with_index(input_dict['extractor'])
                #generated_input = list(input_source.iloc[batch_indexes].apply(apply_func, axis=1).values)
                
                #generated_input = [input_dict['extractor'](generated_input[i], i) for i in range(0, generated_input.shape[0])]
                
                #generated_input = [input_dict['extractor'](input_source.iloc[global_i], i) for i, global_i in enumerate(batch_indexes)]
                
                source_input = input_source.iloc[batch_indexes]
                generated_input = []
                i = 0
                for _, row in source_input.iterrows() :
                    generated_input.append(input_dict['extractor'](row, i))
                    i += 1
                
                
                if input_dict['encoder'] is not None :
                    if isinstance(input_dict['encoder'], (BatchEncoder, SparseBatchEncoder)) :
                        generated_input = input_dict['encoder'](generated_input)
                    else :
                        generated_input = np.concatenate([np.expand_dims(input_dict['encoder'](inp), axis=0) for inp in generated_input], axis=0)
                elif input_dict['transformer'] is not None :
                    if isinstance(input_dict['transformer'], (BatchTransformer, SparseBatchTransformer)) :
                        generated_input = input_dict['transformer'](np.vstack(generated_input))
                    else :
                        generated_input = np.concatenate([np.expand_dims(input_dict['transformer'](inp), axis=0) for inp in generated_input], axis=0)
                else :
                    generated_input = np.vstack(generated_input)
                
                if 'dim' in input_dict :
                    new_dim = tuple([self.batch_size] + list(input_dict['dim']))
                    generated_input = np.reshape(generated_input, new_dim)
                
                generated_inputs.append(generated_input)
            elif input_dict['source_type'] == 'matrix' :
                generated_input = input_source[batch_indexes]
                if self.densify_batch_matrices and isinstance(input_source, (sp.csr_matrix, sp.csc_matrix)) :
                    generated_input = np.array(generated_input.todense())
                
                if input_dict['extractor'] is not None :
                    generated_input = np.vstack([input_dict['extractor'](generated_input[i], i) for i in range(0, generated_input.shape[0])])

                if input_dict['transformer'] is not None :
                    if isinstance(input_dict['transformer'], (BatchTransformer, SparseBatchTransformer)) :
                        generated_input = input_dict['transformer'](generated_input)
                    else :
                        generated_input = np.concatenate([np.expand_dims(input_dict['transformer'](generated_input[i]), axis=0) for i in range(generated_input.shape[0])], axis=0)
                
                if 'dim' in input_dict :
                    new_dim = tuple([self.batch_size] + list(input_dict['dim']))
                    generated_input = np.reshape(generated_input, new_dim)
                
                generated_inputs.append(generated_input)
            else :
                raise NotImplementedError()
        
        #Generate outputs
        if self.outputs is not None :
            generated_outputs = []
            for output_dict in self.outputs :
                if output_dict['source_type'] == 'matrix' :
                    output_source = self.sources[output_dict['source']]
                    
                    generated_output = output_source[batch_indexes]
                    if self.densify_batch_matrices and isinstance(output_source, (sp.csr_matrix, sp.csc_matrix)) :
                        generated_output = np.array(generated_output.todense())
                    
                    if output_dict['extractor'] is not None :
                        generated_output = np.vstack([output_dict['extractor'](generated_output[i], i) for i in range(0, generated_output.shape[0])])

                    if output_dict['transformer'] is not None :
                        if isinstance(output_dict['transformer'], (BatchTransformer, SparseBatchTransformer)) :
                            generated_output = output_dict['transformer'](generated_output)
                        else :
                            generated_output = np.concatenate([np.expand_dims(output_dict['transformer'](generated_output[i]), axis=0) for i in range(generated_output.shape[0])], axis=0)

                    if 'dim' in output_dict :
                        new_dim = tuple([self.batch_size] + list(output_dict['dim']))
                        generated_output = np.reshape(generated_output, new_dim)
                    
                    generated_outputs.append(generated_output)
                elif output_dict['source_type'] == 'dataframe' :
                    output_source = self.sources[output_dict['source']]
                    #generated_output = [output_dict['extractor'](output_source.iloc[global_i], i) for i, global_i in enumerate(batch_indexes)]
                    
                    source_output = output_source.iloc[batch_indexes]
                    generated_output = []
                    i = 0
                    for _, row in source_output.iterrows() :
                        generated_output.append(output_dict['extractor'](row, i))
                        i += 1
                    
                
                    if output_dict['transformer'] is not None :
                        if isinstance(output_dict['transformer'], (BatchTransformer, SparseBatchTransformer)) :
                            generated_output = output_dict['transformer'](np.vstack(generated_output))
                        else :
                            generated_output = np.concatenate([np.expand_dims(output_dict['transformer'](inp), axis=0) for inp in generated_output], axis=0)
                    else :
                        generated_output = np.vstack(generated_output)

                    if 'dim' in output_dict :
                        new_dim = tuple([self.batch_size] + list(output_dict['dim']))
                        generated_output = np.reshape(generated_output, new_dim)
                    
                    generated_outputs.append(generated_output)
                elif output_dict['source_type'] == 'zeros' :
                    if 'dim' in output_dict :
                        new_dim = tuple([self.batch_size] + list(output_dict['dim']))
                        generated_outputs.append(np.zeros(new_dim))
                    else :
                        generated_outputs.append(np.zeros(self.batch_size))
                else :
                    raise NotImplementedError()

        if generated_outputs is not None :
            return generated_inputs, generated_outputs
        
        return generated_inputs

import operator

class MultiDataGenerator(keras.utils.Sequence) :
    
    def __init__(self, data_gens, sampling_factors, reshuffle_flags, epoch_loss_factors, dummy_outputs=True) :
        self.data_gens = data_gens
        
        self.dummy_outputs = dummy_outputs
        
        self.lens = [len(data_gen) for data_gen in data_gens]
        
        self.indexes = np.ones((np.max(self.lens), len(data_gens)), dtype=np.int) * -1
        
        for i in range(len(data_gens)) :
            row_i = 0
            
            #orig_index = np.arange(len(data_gens[i]), dtype=np.int)
            
            n_samples = int(len(data_gens[i]) * sampling_factors[i])
            for jj in range(n_samples) :
                j = jj % len(data_gens[i])
                
                self.indexes[row_i, i] = j
                row_i += 1
            
            if reshuffle_flags[i] :
                np.random.shuffle(self.indexes[:, i])
        
        self.trainable = np.zeros(self.indexes.shape)
        self.trainable[self.indexes != -1] = 1
        self.indexes[self.indexes == -1] = 0
        
        self.epoch = 0
        self.epoch_loss_factors = []
        for i in range(len(data_gens)) :
            loss_factor_dict = epoch_loss_factors[i]
            
            max_epoch = np.max([d_key for d_key in loss_factor_dict])
            self.epoch_loss_factors.append(np.zeros(max_epoch + 1))
            for epoch, factor in sorted(loss_factor_dict.items(), key=operator.itemgetter(0)) :
                self.epoch_loss_factors[-1][epoch:] = factor

    def __len__(self) :
        return np.max(self.lens)

    def __getitem__(self, index):
        index_row = self.indexes[index, :]
        train_row = self.trainable[index, :]
        
        inputs = []
        outputs = []
        
        for i, data_gen in enumerate(self.data_gens) :
            data_gen_index = index_row[i]
            data_gen_train = train_row[i]
            
            train_vec = np.zeros((data_gen.batch_size, 1))
            train_vec[:] = data_gen_train
            
            loss_factor_vec = np.zeros((data_gen.batch_size, 1))
            loss_factor_vec[:] = self.epoch_loss_factors[i][self.epoch if self.epoch < len(self.epoch_loss_factors[i]) else -1]

            data_gen_tup = data_gen[data_gen_index]
            if data_gen.outputs is not None :
                data_gen_input = data_gen_tup[0]
                data_gen_output = data_gen_tup[1]

                inputs.extend(data_gen_input)
                inputs.append(train_vec)
                inputs.append(loss_factor_vec)

                if not self.dummy_outputs or i == 0 :
                    outputs.extend(data_gen_output)
            else :
                data_gen_input = data_gen_tup

                inputs.extend(data_gen_input)
                inputs.append(train_vec)
                inputs.append(loss_factor_vec)
        
        if len(outputs) != 0 :
            return inputs, outputs
        return inputs

    def on_epoch_end(self) :
        self.epoch += 1
        
        for i in range(len(self.data_gens)) :
            self.data_gens[i].on_epoch_end()


def get_bellcurve_shifter() :
    shift_range = (np.arange(71, dtype=np.int) - 35)
    shift_probs = np.zeros(shift_range.shape[0])
    shift_probs[:] = 0.1 / float(shift_range.shape[0] - 11)
    shift_probs[int(shift_range.shape[0]/2)] = 0.5
    shift_probs[int(shift_range.shape[0]/2)-5:int(shift_range.shape[0]/2)] = 0.2 / 5.
    shift_probs[int(shift_range.shape[0]/2)+1:int(shift_range.shape[0]/2)+1+5] = 0.2 / 5.
    shift_probs /= np.sum(shift_probs)
    
    return PositionShifter(shift_range, shift_probs)
