# utils/mpi_context.py

class MPIContext:
  """
  Centralized MPI context manager
  Initialize once at program entry
  """
  _instance = None
  
  def __init__(self):
    self._initialize_mpi()
  
  def _initialize_mpi(self):
    """ Try to initialize MPI, detect if running in parallel """
    try:
      from mpi4py import MPI
      self.comm = MPI.COMM_WORLD
      self.rank = self.comm.rank
      self.size = self.comm.size
      self.available = True
      self.is_parallel = self.size > 1
      self.MPI = MPI
    except (ImportError, AttributeError):
      # Fallback to serial (MPI not available)
      self.comm = None
      self.rank = 0
      self.size = 1
      self.available = False
      self.is_parallel = False
      self.MPI = None
      
  @classmethod
  def get_instance(cls):
    """Singleton pattern - ensures MPI is initialized only once."""
    if cls._instance is None:
      cls._instance = cls()
    return cls._instance
    
  def barrier(self):
    """ Safe barrier - only calls if MPI is available """
    if self.comm is not None:
      self.comm.Barrier()
      
  def bcast(self,obj,root=0):
    """ Safe broadcast - only calls if MPI is available """
    if self.comm is not None:
      return self.comm.bcast(obj, root=root)
    return obj
    
  def allreduce(self,value,op):
    """
    Safe allreduce - only calls if MPI is available.
    In serial mode, returns the value unchanged.
    
    Parameters:
    -----------
    value : scalar
        Value to reduce (same type on all ranks)
    op : MPI operation
        Reduction operation (e.g., MPI.MIN, MPI.MAX, MPI.SUM)
    
    Returns:
    --------
    result : scalar
        Reduced value (same on all ranks)
    """
    if self.comm is not None and self.MPI is not None:
      return self.comm.allreduce(value, op=op)
    return value