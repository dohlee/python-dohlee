import pysam
import numpy as np

class Variant(object):
    """
    Base class for variants. Simply encapsulates pysam.VariantRecord object, 
    but deals with features like VAFs, or ANN fields (if exists) more conveniently.

    Supported annotation type: snpeff
    """
    def __init__(self, e, annotation_type):
        self.alleles = e.alleles
        self.alts = e.alts
        self.chrom = e.chrom
        self.contig = e.contig
        self.filter = e.filter
        self.format = e.format
        self.id = e.id
        self.info = e.info
        self.pos = e.pos
        self.qual = e.qual
        self.ref = e.ref
        self.rid = e.rid
        self.rlen = e.rlen
        self.samples = e.samples
        self.start = e.start
        self.stop = e.stop
        self.annotation_type = annotation_type

        if annotation_type == 'snpeff':
            self.annotations = [SnpEffAnnotation(s) for s in self.info['ANN']]

    def get_refallele(self):
        return self.ref 

    def get_altallele(self):
        """Return alt allele.
        Note that this method returns the leftmost alt alleles of the entry.
        """
        return self.alts[0]

    def get_normal_refcount(self):
        raise NotImplementedError

    def get_normal_altcount(self):
        raise NotImplementedError

    def get_tumor_refcount(self):
        raise NotImplementedError

    def get_tumor_altcount(self):
        raise NotImplementedError

    def get_normal_vaf(self):
        num_normalref, num_normalalt = self.get_normal_refcount(), self.get_normal_altcount()
        num_total = num_normalref + num_normalalt
        if num_total == 0:
            return np.nan
        else:
            return num_normalalt / (num_normalref + num_normalalt)

    def get_tumor_vaf(self):
        num_tumorref, num_tumoralt = self.get_tumor_refcount(), self.get_tumor_altcount()
        num_total = num_tumorref + num_tumoralt
        if num_total == 0:
            return np.nan
        else:
            return num_tumoralt / (num_tumorref + num_tumoralt)

    def filter_annotation(self, func):
        """Only keep annotation that meet the criterion.
        """
        self.annotations = list(filter(func, self.annotations))
        return self.annotations
    
    def __str__(self):
        r, a = self.get_refallele(), self.get_altallele()
        if self.annotation_type is None:
            return f'{self.chrom}:{self.pos} ({r}->{a})'
        elif self.annotation_type == 'snpeff':
            anns = ','.join([ann.gene_symbol + '[' + ann.hgvs_p + ']' for ann in self.annotations])
            return f'{self.chrom}:{self.pos} ({r}->{a}, {anns})'

class StrelkaSomaticVariant(Variant):
    
    def __init__(self, e, annotation_type=None):
        super(StrelkaSomaticVariant, self).__init__(e, annotation_type)

    def get_normal_refcount(self):
        ref = self.get_refallele()
        return self.samples['NORMAL'][f'{ref}U'][0]
    
    def get_normal_altcount(self):
        alt = self.get_altallele()
        return self.samples['NORMAL'][f'{alt}U'][0]
    
    def get_tumor_refcount(self):
        ref = self.get_refallele()
        return self.samples['TUMOR'][f'{ref}U'][0]

    def get_tumor_altcount(self):
        alt = self.get_altallele()
        return self.samples['TUMOR'][f'{alt}U'][0]

class VarscanSomaticVariant(Variant):
    
    def __init__(self, e, annotation_type=None):
        super(VarscanSomaticVariant, self).__init__(e, annotation_type)
        self.somatic_p = e.info['SPV']
        self.germline_p = e.info['GPV']
        self.somatic_status = e.info['SS']

    def get_normal_refcount(self):
        ref = self.get_refallele()
        return self.samples['NORMAL']['RD']
    
    def get_normal_altcount(self):
        alt = self.get_altallele()
        return self.samples['NORMAL']['AD']
    
    def get_tumor_refcount(self):
        ref = self.get_refallele()
        return self.samples['TUMOR']['RD']

    def get_tumor_altcount(self):
        alt = self.get_altallele()
        return self.samples['TUMOR']['AD']

class SnpEffAnnotation(object):
    def __init__(self, annotation_string):
        tokens = annotation_string.split('|')
        
        self.allele = tokens[0]
        self.annotation = tokens[1].split('&')
        self.impact = tokens[2]
        self.gene_symbol = tokens[3]
        self.gene_id = tokens[4]
        self.feature_type = tokens[5]
        self.feature_id = tokens[6]
        self.transcript_biotype = tokens[7]
        self.rank, self.total = map(int, tokens[8].split('/')) if tokens[8] != '' else [0, 0]
        self.hgvs_c = tokens[9]
        self.hgvs_p = tokens[10]
        self.cDNA_pos, self.cDNA_len = map(int, tokens[11].split('/')) if tokens[11] != '' else [0, 0]
        self.CDS_pos, self.CDS_len = map(int, tokens[12].split('/')) if tokens[12] != '' else [0, 0]
        self.protein_pos, self.protein_len = map(int, tokens[13].split('/')) if tokens[13] != '' else [0, 0]
        self.distance_to_feature = int(tokens[14]) if tokens[14] != '' else 0
    
    def __str__(self):
        return 'Variant Annotation\nAllele: %s\nAnnotation: %s\nPutative impact: %s\nGene: %s(%s)\nFeature: %s(%s)\nTranscript type: %s\nDNA-level variant: %s\nProtein-level variant %s\n' \
                % (self.allele, self.annotation, self.impact, self.gene_symbol, self.gene_id, self.feature_type, self.feature_id, self.transcript_biotype, self.hgvs_c, self.hgvs_p)

class VCF(object):
    """
    Base class for Variant Call Format (VCF).
    """
    def __init__(self, fp, annotation_type):
        self.fp = fp
        self.vcf = pysam.VariantFile(fp)
        self.annotation_type = annotation_type
        self.variant = None

    def fetch(self, chrom=None, start=None, end=None, region=None, filter=None):
        """Yields VCF entries lying between given genomic region.
        Without chrom or region, yields all entries.
        """
        for entry in self.vcf.fetch(reference=chrom, start=start, end=end, region=region):
            if filter:
                if entry.filter.keys()[0] == filter:
                    yield self.variant(entry, self.annotation_type)
            else:
                yield self.variant(entry, self.annotation_type)
    
    def fetch_feature(self, feature_id, chrom=None, start=None, end=None, region=None, filter=None):
        """Fetch variants associated with feature_id. Utilizes info.ANN field to fetch specific feature.
        Passing rough genomic region may significantly reduce running time.
        """
        for entry in self.vcf.fetch(reference=chrom, start=start, end=end, region=region):
            if filter and entry.filter.keys()[0] != filter:
                continue
            
            v = self.variant(entry, self.annotation_type)
            annotations = v.annotations 
            if any(ann.feature_id.startswith(feature_id) for ann in annotations):
                v.filter_annotation(func=lambda ann: ann.feature_id.startswith(feature_id))
                yield v

class StrelkaSomaticVCF(VCF):
    """
    Handles somatic VCF file generated by Strelka2.
    """
    def __init__(self, fp, annotation_type):
        super(StrelkaSomaticVCF, self).__init__(fp, annotation_type)
        self.variant = StrelkaSomaticVariant

class VarscanSomaticVCF(VCF):
    """
    Handles somatic VCF file generated by VarScan.
    """
    def __init__(self, fp, annotation_type):
        super(VarscanSomaticVCF, self).__init__(fp, annotation_type)
        self.variant = VarscanSomaticVariant
